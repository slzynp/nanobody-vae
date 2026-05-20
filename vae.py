import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import secrets
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_LEN    = 80   # interior torsions for merged FW3+CDR3+FW4 region
N_ANGLES   = 2    # phi, psi  (radians)
HIDDEN_DIM = 64
LATENT_DIM = 32


# ============================================================================
# VAE MODEL  —  Gaussian prior + posterior, Von Mises likelihood
# ============================================================================

class NanobodyVAE(nn.Module):
    """Recurrent VAE for FW3+CDR3+FW4 torsion angles.

    Prior:      p(z)   = N(0, I)
    Posterior:  q(z|x) = N(mu(x), diag(sigma²(x)))
    Likelihood: p(x|z) = VM(mu_x(z), kappa_x(z))

    ELBO = E_q[log p(x|z) / T] - beta * KL(q||p)

    where T is the sequence length (per-residue normalisation) and
    beta is annealed from 1e-4 to max_beta over training.
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
        self.fc_out       = nn.Linear(hidden_dim, input_dim)   # angle means
        self.fc_kappa_out = nn.Linear(hidden_dim, input_dim)   # concentrations

    # ------------------------------------------------------------------ #

    def encode(self, x, seq_lengths):
        """x → (mu_z, log_var_z)  shapes: (batch, latent_dim)"""
        packed = pack_padded_sequence(x, seq_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h = self.encoder_rnn(packed)
        h    = h.squeeze(0)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z, seq_len):
        """z → (mu_x, kappa_x)  shapes: (batch, seq_len, input_dim)"""
        batch      = z.shape[0]
        z_expand   = z.unsqueeze(1).expand(batch, seq_len, self.latent_dim)
        rnn_out, _ = self.decoder_rnn(z_expand)

        raw     = self.fc_out(rnn_out)
        mu_x    = torch.atan2(torch.sin(raw), torch.cos(raw))   # wrap to [-pi, pi]
        kappa_x = F.softplus(self.fc_kappa_out(rnn_out)) + 1e-4
        return mu_x, kappa_x

    # ------------------------------------------------------------------ #

    def elbo_loss(self, x, seq_lengths, beta):
        """Negative ELBO with beta-annealed KL.

        Returns scalar (per-sample average), minimise this.

        Loss = -E_q[log p(x|z) / T] + beta * KL(q||p)
        """
        _, seq_len, _ = x.shape

        # Encode → posterior params
        mu_z, log_var = self.encode(x, seq_lengths)

        # Reparameterisation trick
        std = (0.5 * log_var).exp()
        z   = mu_z + std * torch.randn_like(std)

        # Decode → Von Mises params
        mu_x, kappa_x = self.decode(z, seq_len)

        # Reconstruction: Von Mises log-likelihood, per-residue normalised
        mask  = (torch.arange(seq_len, device=x.device)[None, :]
                 < seq_lengths[:, None]).float().unsqueeze(-1)  # (B, T, 1)
        recon = (dist.VonMises(mu_x, kappa_x).log_prob(x) * mask).sum(dim=[1, 2])
        recon = recon / seq_lengths.float()                      # per-residue

        kl = -0.5 * (1 + log_var - mu_z.pow(2) - log_var.exp()).sum(-1)  # (B,)

        return (-recon + beta * kl).mean()

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def eval_loss(self, x, seq_lengths, beta):
        """ELBO loss in eval mode (no reparameterisation noise)."""
        was_training = self.training
        self.eval()
        loss = self.elbo_loss(x, seq_lengths, beta)
        if was_training:
            self.train()
        return loss.item()

    def sample(self, num_samples=50, seq_len=MAX_LEN):
        """Sample from Gaussian prior and decode."""
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            mu_x, _ = self.decode(z, seq_len)
        return z.cpu(), mu_x.cpu()


# ============================================================================
# BETA-ANNEALING SCHEDULE
# ============================================================================

def get_beta_schedule(epoch, warmup_epochs=70, anneal_end=200, max_beta=0.01):
    """β-annealing: warm-up → gradual increase → plateau.

    Warmup holds beta at 1e-4 so the encoder freely uses the latent space
    before KL pressure starts. Beta then ramps linearly to max_beta, which
    is kept low (0.01) so KL stays smaller than the reconstruction term.
    """
    if epoch < warmup_epochs:
        return 1e-4
    elif epoch < anneal_end:
        progress = (epoch - warmup_epochs) / (anneal_end - warmup_epochs)
        return 1e-4 + progress * (max_beta - 1e-4)
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
        # torsions column: [[phi, psi, omega], ...] degrees → phi/psi only, radians
        self.angles = self.data['torsions'].apply(
            lambda x: np.deg2rad(np.array(eval(x), dtype=np.float32)[:, :2])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.angles[idx],

    def create_train_val_split(self, val_fraction=0.2, random_seed=42):
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
              epochs=200, device='cpu', patience=20,
              train_indices=None, val_indices=None,
              warmup_epochs=70, anneal_end=200, max_beta=0.01,
              lr=5e-4, weight_decay=1e-5, save_dir='.'):

    print(f"\n{'='*70}")
    print("TRAINING NANOBODY VAE  (beta-annealed ELBO, Von Mises likelihood)")
    print(f"{'='*70}\n")

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses, kl_values, beta_values = [], [], [], []

    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0

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

            optimizer.zero_grad()
            loss = vae.elbo_loss(x, seq_lengths, beta)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            n_train    += 1

        train_loss /= n_train

        # ─── VALIDATION ───
        vae.eval()
        val_loss = 0.0
        val_kl   = 0.0
        n_val    = 0

        with torch.no_grad():
            for x, seq_lengths in val_loader:
                x           = x.to(device)
                seq_lengths = seq_lengths.to(device)

                val_loss += vae.elbo_loss(x, seq_lengths, beta).item()

                # Closed-form KL for monitoring (no free bits, no beta)
                mu_z, log_var_z = vae.encode(x, seq_lengths)
                kl = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(-1).mean()
                val_kl += kl.item()
                n_val  += 1

        val_loss /= n_val
        val_kl   /= n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        kl_values.append(val_kl)

        # ─── EARLY STOPPING ───
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                'model':         vae.state_dict(),
                'epoch':         epoch,
                'train_losses':  train_losses,
                'val_losses':    val_losses,
                'train_indices': train_indices,
                'val_indices':   val_indices,
            }, os.path.join(save_dir, 'vae_checkpoint.pt'))
            print(f"Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"kl={val_kl:.4f}, β={beta:.4f} ✓")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"kl={val_kl:.4f}, β={beta:.4f}")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best model saved: {os.path.join(save_dir, 'vae_checkpoint.pt')}")
    print(f"  Best epoch      : {best_epoch+1}")
    print(f"  Best val loss   : {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return train_losses, val_losses, kl_values, beta_values


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_results(train_losses, val_losses, kl_values, beta_values,
                          save_path="training_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(train_losses) + 1)

    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs, val_losses,   'r-', label='Val',   linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss (ELBO)')
    axes[0, 0].set_title('Training & Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, kl_values, 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence (Validation)'); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, beta_values, color='purple', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('β')
    axes[1, 0].set_title('β-Annealing Schedule'); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, val_losses, 'r-', linewidth=2)
    axes[1, 1].fill_between(epochs, val_losses, alpha=0.2, color='red')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation Loss (Detailed)'); axes[1, 1].grid(alpha=0.3)

    best_epoch = int(np.argmin(val_losses)) + 1
    for ax in axes.flat:
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Best (epoch {best_epoch})')

    plt.suptitle('Nanobody VAE — free-bits ELBO, Von Mises likelihood', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    latent_dim    = LATENT_DIM
    hidden_dim    = HIDDEN_DIM
    warmup_epochs = 50
    anneal_end    = 170
    max_beta      = 0.01
    epochs        = 300
    patience      = 20
    batch_size    = 32
    lr            = 5e-4
    weight_decay  = 1e-5

    timestamp = datetime.now().strftime('%Y%m%d')
    run_name  = f"run_{timestamp}_{secrets.token_hex(3)}"
    save_dir  = os.path.join('experiments', run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Run: {run_name}\n")

    config = {
        'latent_dim':    latent_dim,
        'hidden_dim':    hidden_dim,
        'warmup_epochs': warmup_epochs,
        'anneal_end':    anneal_end,
        'max_beta':      max_beta,
        'epochs':        epochs,
        'patience':      patience,
        'batch_size':    batch_size,
        'lr':            lr,
        'weight_decay':  weight_decay,
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
    print(f"  Posterior  : N(mu, sigma²)  ({latent_dim}D)")
    print(f"  Likelihood : VM(mu_x, kappa_x)")
    print(f"  Params     : {sum(p.numel() for p in vae.parameters()):,}\n")

    train_losses, val_losses, kl_values, beta_values = train_vae(
        vae, train_loader, val_loader,
        epochs=epochs, device=device, patience=patience,
        train_indices=train_indices, val_indices=val_indices,
        warmup_epochs=warmup_epochs, anneal_end=anneal_end, max_beta=max_beta,
        lr=lr, weight_decay=weight_decay,
        save_dir=save_dir,
    )

    plot_training_results(
        train_losses, val_losses, kl_values, beta_values,
        save_path=os.path.join(save_dir, 'training_curves.png'),
    )


if __name__ == "__main__":
    main()
