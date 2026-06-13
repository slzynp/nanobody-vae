import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import secrets
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import pyro
import pyro.distributions as pyrodist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import zuko
from pyro.contrib.zuko import ZukoToPyro

# ============================================================================
# CONSTANTS
# ============================================================================

MAX_LEN    = 80
N_ANGLES   = 2
HIDDEN_DIM = 64
LATENT_DIM = 32
MIN_KAPPA = 5.0  # minimum concentration for Von Mises distribution
MAX_KAPPA = 50.0 # maximum concentration for Von Mises distribution
SEED = 42
# ============================================================================
# VAE MODEL  —  MAF prior + Gaussian posterior, Von Mises likelihood
# ============================================================================

class NanobodyVAE(nn.Module):
    """Recurrent VAE for FW3+CDR3+FW4 torsion angles.

    Prior:      p(z)   = N(0, I)
    Posterior:  q(z|x) = N(mu(x), diag(sigma²(x)))
    Likelihood: p(x|z) = VM(mu_x(z), kappa_x(z))

    ELBO = E_q[log p(x|z) / T] - beta * KL(q||p)

    where T is the sequence length (per-residue normalisation) and
    beta is annealed from 0.0001 to max_beta over training.
    """

    def __init__(self, input_dim=N_ANGLES, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, device='cpu'):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device     = device

        # ENCODER: angle sequence → Gaussian posterior parameters
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu       = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var  = nn.Linear(hidden_dim, latent_dim)

        # DECODER: latent code → Von Mises likelihood parameters
        self.decoder_rnn  = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.fc_out       = nn.Linear(hidden_dim, input_dim)
        self.fc_kappa_out = nn.Linear(hidden_dim, input_dim)

        # PRIOR: MAF flow p(z), PYRO's implementation 
        self.prior = zuko.flows.MAF(
            features=latent_dim,
            hidden_features=[hidden_dim, hidden_dim],
            transforms=3,
        ).to(device)
        
    def encode(self, x, seq_lengths):
        packed = pack_padded_sequence(x, seq_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h = self.encoder_rnn(packed)
        h    = h.squeeze(0) 
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z, seq_len):
        batch      = z.shape[0]
        z_expand   = z.unsqueeze(1).expand(batch, seq_len, self.latent_dim)
        rnn_out, _ = self.decoder_rnn(z_expand)

        raw     = self.fc_out(rnn_out)
        mu_x    = torch.atan2(torch.sin(raw), torch.cos(raw))
        kappa_x = MIN_KAPPA + (MAX_KAPPA - MIN_KAPPA) * torch.sigmoid(self.fc_kappa_out(rnn_out))
        return mu_x, kappa_x

    def model(self, x, seq_lengths, beta):
        pyro.module("nanobody_vae", self)
        batch_size, seq_len, _ = x.shape

        with pyro.plate("data", batch_size): ## independence over sequences
            with pyro.poutine.scale(scale=beta): ## anneal KL term 
                z = pyro.sample("z", ZukoToPyro(self.prior())) ## sample z from the MAF prior flow 

            mu_x, kappa_x = self.decode(z, seq_len)
            mask = (torch.arange(seq_len, device=x.device)[None, :]
                    < seq_lengths[:, None]).float().unsqueeze(-1)
            log_lik = (
                dist.VonMises(mu_x, kappa_x).log_prob(x) * mask
            ).sum(dim=[1, 2]) / seq_lengths.float()
            pyro.factor("x_obs", log_lik)

    def guide(self, x, seq_lengths, beta):
        pyro.module("nanobody_vae", self)
        mu_z, log_var = self.encode(x, seq_lengths)
        std = (0.5 * log_var).exp()

        with pyro.plate("data", x.shape[0]):
            with pyro.poutine.scale(scale=beta):
                pyro.sample("z", pyrodist.Normal(mu_z, std).to_event(1))

    @torch.no_grad()
    def reconstruction_loss(self, x, seq_lengths):
        """Mean Von Mises NLL at posterior mean z (monitoring only)."""
        mu_z, _ = self.encode(x, seq_lengths)
        seq_len = x.size(1)
        mu_x, kappa_x = self.decode(mu_z, seq_len)
        mask = (torch.arange(seq_len, device=x.device)[None, :]
                < seq_lengths[:, None]).float().unsqueeze(-1)
        nll = -(dist.VonMises(mu_x, kappa_x).log_prob(x) * mask).sum(dim=[1, 2]) / seq_lengths.float()
        return nll.mean().item()

    @torch.no_grad()
    def generate(self, num_samples=78, seq_len=MAX_LEN, stochastic=True):
        """Sample z ~ N(0, I) via the Pyro prior and decode to angle sequences.

        Args:
            num_samples: number of sequences to generate
            seq_len:     residues per sequence
            stochastic:  if True, draw angles from VM(mu_x, kappa_x);
                         if False, return the decoder mean mu_x directly

        Returns:
            z:      (num_samples, latent_dim) — latent codes
            angles: (num_samples, seq_len, 2) — φ/ψ in radians
        """
        self.eval()
        with pyro.plate("samples", num_samples):
            z = pyro.sample("z_prior", ZukoToPyro(self.prior()))
        mu_x, kappa_x = self.decode(z, seq_len)
        angles = pyrodist.VonMises(mu_x, kappa_x).sample() if stochastic else mu_x
        return z.cpu(), angles.cpu()


# ============================================================================
# BETA-ANNEALING SCHEDULE
# ============================================================================

def get_beta_schedule(epoch, warmup_epochs=70, anneal_end=200, max_beta=0.1):
    """beta-annealing: warm-up → gradual increase → plateau."""
    if epoch < warmup_epochs:
        return 1e-3   
    elif epoch < anneal_end:
        progress = (epoch - warmup_epochs) / (anneal_end - warmup_epochs)
        return 1e-3 + progress * (max_beta - 1e-3)
    else:
        return max_beta


# ============================================================================
# DATASET / COLLATION
# ============================================================================

def collate_fn(batch):
    sequences, lengths = [], []
    for (angles,) in batch:
        lengths.append(len(angles))
        sequences.append(torch.tensor(angles, dtype=torch.float32))

    lengths = torch.tensor(lengths, dtype=torch.long)
    padded  = torch.zeros(len(batch), MAX_LEN, sequences[0].size(-1))
    for i, seq in enumerate(sequences):
        padded[i, :lengths[i]] = seq[:lengths[i]]
    return padded, lengths


class NanobodyDataset(torch.utils.data.Dataset):
    def __init__(self, file="nanobodies_filtered.csv"):
        self.data = pd.read_csv(file)
        self.angles = self.data['torsions'].apply(
            lambda x: np.deg2rad(np.array(eval(x), dtype=np.float32)[:, :2])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.angles[idx],

    def create_train_val_split(self, val_fraction=0.2, random_seed=SEED):
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.data))
        split   = int(len(self.data) * (1 - val_fraction))
        return indices[:split], indices[split:]

    def get_dataloader(self, indices, batch_size=32, shuffle=True):
        subset = torch.utils.data.Subset(self, indices)
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_vae(vae, train_loader, val_loader,
              epochs=200, device='cpu',
              train_indices=None, val_indices=None,
              warmup_epochs=70, anneal_end=200, max_beta=0.01,
              lr=5e-4, weight_decay=1e-5, save_dir='.'):

    print(f"\n{'='*70}")
    print("TRAINING NANOBODY VAE  (SVI, Trace_ELBO, Von Mises likelihood)")
    print(f"{'='*70}\n")

    pyro.clear_param_store()

    optimizer = ClippedAdam({
        "lr":           lr,
        "weight_decay": weight_decay,
        "clip_norm":    10.0,   # gradient clipping important for RNNs
    })
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_losses, val_losses, kl_values, beta_values, recon_losses = [], [], [], [], []

    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(epochs):
        beta = get_beta_schedule(epoch, warmup_epochs=warmup_epochs,
                                 anneal_end=anneal_end, max_beta=max_beta)
        beta_values.append(beta)

        # ─── TRAINING ───
        vae.train()
        train_loss = 0.0
        n_train    = 0

        for x, seq_lengths in train_loader:
            x           = x.to(device)
            seq_lengths = seq_lengths.to(device)

            # svi.step runs forward + backward + optimizer step, returns loss sum
            loss = svi.step(x, seq_lengths, beta)
            train_loss += loss / x.size(0)
            n_train    += 1

        train_loss /= n_train

        # ─── VALIDATION ───
        vae.eval()
        val_loss = 0.0
        val_kl   = 0.0
        n_val    = 0

        for x, seq_lengths in val_loader:
            x           = x.to(device)
            seq_lengths = seq_lengths.to(device)

            # evaluate_loss computes ELBO without a gradient step
            val_loss += svi.evaluate_loss(x, seq_lengths, beta) / x.size(0)

            with torch.no_grad():
                mu_z, log_var_z = vae.encode(x, seq_lengths)
                kl = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(-1).mean()
                val_kl += kl.item()

            n_val += 1

        val_loss /= n_val
        val_kl   /= n_val

        val_recon = sum(
            vae.reconstruction_loss(x.to(device), sl.to(device))
            for x, sl in val_loader
        ) / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        kl_values.append(val_kl)
        recon_losses.append(val_recon)

        # ─── CHECKPOINT ───
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                'model':         vae.state_dict(),
                'epoch':         epoch,
                'train_losses':  train_losses,
                'val_losses':    val_losses,
                'train_indices': train_indices,
                'val_indices':   val_indices,
            }, os.path.join(save_dir, 'vae_checkpoint.pt'))
            print(f"Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"kl={val_kl:.4f}, beta={beta:.4f} *")
        else:
            print(f"Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"kl={val_kl:.4f}, beta={beta:.4f}")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best model saved: {os.path.join(save_dir, 'vae_checkpoint.pt')}")
    print(f"  Best epoch      : {best_epoch+1}")
    print(f"  Best val loss   : {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return train_losses, val_losses, kl_values, beta_values, recon_losses


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_results(train_losses, val_losses, kl_values, beta_values, recon_losses,
                          save_path="training_curves.png", max_epochs=3000):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(train_losses) + 1)
    best_epoch = int(np.argmin(val_losses)) + 1

    axes[0, 0].plot(epochs, recon_losses, color='darkorange', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('NLL (nats/residue)')
    axes[0, 0].set_title('Reconstruction Loss (Validation)')
    axes[0, 0].set_xlim(1, best_epoch + 20)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, kl_values, 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence (Validation)')
    axes[0, 1].set_xlim(1, best_epoch + 20)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, beta_values, color='purple', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('beta')
    axes[1, 0].set_title('beta-Annealing Schedule')
    axes[1, 0].set_xlim(1, best_epoch + 20)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, val_losses, 'r-', linewidth=2)
    axes[1, 1].fill_between(epochs, val_losses, alpha=0.2, color='red')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation Loss (ELBO)')
    axes[1, 1].set_xlim(1, best_epoch + 20)
    axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Best (epoch {best_epoch})')

    plt.suptitle('Nanobody VAE Training Curves with MAF Prior' ,
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    latent_dim    = LATENT_DIM
    hidden_dim    = HIDDEN_DIM
    warmup_epochs = 200
    anneal_end    = 1000
    max_beta      = 0.1
    epochs        = 3000
    batch_size    = 32
    lr            = 5e-4
    weight_decay  = 1e-5
    min_kappa      = MIN_KAPPA
    max_kappa      = MAX_KAPPA
    seed         = SEED
    timestamp = datetime.now().strftime('%Y%m%d')
    run_name  = f"run_{timestamp}_{secrets.token_hex(3)}"
    save_dir  = os.path.join('experiments', run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Run: {run_name}\n")

    config = {
        'vae_type':      'svi',
        'latent_dim':    latent_dim,
        'hidden_dim':    hidden_dim,
        'warmup_epochs': warmup_epochs,
        'anneal_end':    anneal_end,
        'max_beta':      max_beta,
        'epochs':        epochs,
        'batch_size':    batch_size,
        'lr':            lr,
        'weight_decay':  weight_decay,
        'min_kappa':     min_kappa,
        'max_kappa':     max_kappa,
        'seed':         seed,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    open(os.path.join(save_dir, 'notes.txt'), 'w').close()

    dataset = NanobodyDataset()
    train_indices, val_indices = dataset.create_train_val_split()
    train_loader = dataset.get_dataloader(train_indices, batch_size=batch_size, shuffle=True)
    val_loader   = dataset.get_dataloader(val_indices,   batch_size=batch_size, shuffle=False)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}\n")

    vae = NanobodyVAE(
        input_dim=N_ANGLES, hidden_dim=hidden_dim,
        latent_dim=latent_dim, device=device
    ).to(device)

    print(f"NanobodyVAE")
    print(f"  Prior      : N(0, I)  ({latent_dim}D)")
    print(f"  Posterior  : N(mu, sigma^2)  ({latent_dim}D)")
    print(f"  Likelihood : VM(mu_x, kappa_x)")
    print(f"  Optimizer  : ClippedAdam (SVI)")
    print(f"  Params     : {sum(p.numel() for p in vae.parameters()):,}\n")

    train_losses, val_losses, kl_values, beta_values, recon_losses = train_vae(
        vae, train_loader, val_loader,
        epochs=epochs, device=device,
        train_indices=train_indices, val_indices=val_indices,
        warmup_epochs=warmup_epochs, anneal_end=anneal_end, max_beta=max_beta,
        lr=lr, weight_decay=weight_decay,
        save_dir=save_dir,
    )

    plot_training_results(
        train_losses, val_losses, kl_values, beta_values, recon_losses,
        save_path=os.path.join(save_dir, 'training_curves.png'),
        max_epochs=epochs,
    )


if __name__ == "__main__":
    main()
