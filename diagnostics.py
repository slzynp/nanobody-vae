"""
VAE Diagnostics
===============
Checks two known symptoms:
  1. Posterior collapse  — KL ≈ 0, latent dims not used by encoder
  2. Val < Train loss    — overfitting or train/eval mode discrepancy

Run after training:  python diagnostics.py

Outputs:
  diag_kl_per_dim.png                    per-dimension KL and active-unit count
  diag_aggregate_posterior.png           aggregate posterior vs N(0,1) prior (hole diagnosis)
  diag_loss_modes.png                    train vs val loss in train-mode vs eval-mode
  diag_reconstruction_ramachandran.png   real vs reconstructed φ/ψ Ramachandran density
  diag_generated_ramachandran.png        real vs prior-sampled generated φ/ψ Ramachandran density
  diag_report.txt                        numerical summary
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import pyro
from pyro.infer import Trace_ELBO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, norm as scipy_norm
from matplotlib.colors import LogNorm

from vae_svi import (
    NanobodyVAE, NanobodyDataset,
    N_ANGLES, HIDDEN_DIM, LATENT_DIM,
    MIN_KAPPA, MAX_KAPPA,
    collate_fn, get_beta_schedule,
)

CSV_PATH    = "nanobodies_filtered.csv"
OUT_DIR     = None   # set in main() from the run directory
ACTIVE_UNIT_THRESHOLD = 0.1   # nats; standard in the literature

def out(name):
    return os.path.join(OUT_DIR, name)


def _latest_run():
    """Find the latest run directory in experiments/ by modified time, containing a checkpoint."""
    exp_dir = 'experiments'
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError("No 'experiments/' directory. Run vae.py first.")
    runs = [
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
        and os.path.exists(os.path.join(exp_dir, d, 'vae_checkpoint.pt'))
    ]
    if not runs:
        raise FileNotFoundError("No checkpoint found in experiments/. Run vae.py first.")
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(exp_dir, d)))
    return runs[-1]


# ─── Load model ──────────────────────────────────────────────────────────────

def load_model(device, checkpoint_path):
    vae = NanobodyVAE(
        input_dim=N_ANGLES, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, device=device
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model"])

    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?') + 1})")
    return vae, ckpt


# ─── Forward pass helpers ────────────────────────────────────────────────────

@torch.no_grad()
def collect_latents(vae, loader, device, beta):
    """Return mu_z and log_var_z by running the Pyro guide through poutine.trace."""
    vae.eval()
    all_mu, all_lv = [], []
    for x, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        guide_trace = pyro.poutine.trace(vae.guide).get_trace(x, lengths, beta)
        z_fn = guide_trace.nodes["z"]["fn"]   # Independent(Normal(mu_z, std), 1)
        base  = z_fn.base_dist                # Normal(mu_z, std)
        all_mu.append(base.loc.detach().cpu())
        all_lv.append((2 * base.scale.log()).detach().cpu())
    return torch.cat(all_mu), torch.cat(all_lv)


@torch.no_grad()
def compute_elbo_loss(vae, loader, device, beta, train_mode):
    """Negative ELBO (Trace_ELBO) in train or eval mode."""
    elbo = Trace_ELBO()
    if train_mode:
        vae.train()
    else:
        vae.eval()
    losses = []
    for x, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        losses.append(elbo.loss(vae.model, vae.guide, x, lengths, beta))
    return float(np.mean(losses))


# ─── Plot 1: KL per latent dimension ─────────────────────────────────────────

def plot_kl_per_dim(mu_z, lv_z):
    kl_per_dim = 0.5 * (mu_z.pow(2) + lv_z.exp() - 1 - lv_z)
    kl_mean    = kl_per_dim.mean(0).numpy()
    kl_std     = kl_per_dim.std(0).numpy()
    mu_var     = mu_z.var(0).numpy()

    active = (kl_mean > ACTIVE_UNIT_THRESHOLD).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Posterior Collapse Check  —  Active units: {active}/{LATENT_DIM}"
                 f"  (threshold {ACTIVE_UNIT_THRESHOLD} nats)",
                 fontsize=12, fontweight="bold")

    dims = np.arange(LATENT_DIM)

    ax = axes[0]
    colors = ["#e74c3c" if k < ACTIVE_UNIT_THRESHOLD else "#2ecc71" for k in kl_mean]
    ax.bar(dims, kl_mean, color=colors, alpha=0.85)
    ax.errorbar(dims, kl_mean, yerr=kl_std, fmt='none', c='black', capsize=3, lw=1)
    ax.axhline(ACTIVE_UNIT_THRESHOLD, color="orange", lw=1.5, ls="--",
               label=f"Active threshold ({ACTIVE_UNIT_THRESHOLD} nats)")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title("KL per dimension  (red = collapsed, green = active)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(dims, mu_var, color="#3498db", alpha=0.85)
    ax.axhline(0.01, color="orange", lw=1.5, ls="--", label="δ = 0.01")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Var(μ_z) across dataset")
    ax.set_title("Encoder mean variance per dimension\n(near-zero → dim unused)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out("diag_kl_per_dim.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 1] Active units: {active}/{LATENT_DIM}")
    return active, kl_mean


# ─── Plot 2: Reconstruction Ramachandran ─────────────────────────────────────

@torch.no_grad()
def plot_reconstruction_ramachandran(vae, val_loader, device, beta):
    """Compare real vs reconstructed φ/ψ as Ramachandran density plots.

    Uses the Pyro guide trace to get the posterior mean (μ_z) as the
    latent code for reconstruction, matching how the ELBO is evaluated.
    """
    vae.eval()
    real_phi, real_psi, pred_phi, pred_psi = [], [], [], []

    for x, lengths in val_loader:
        x, lengths = x.to(device), lengths.to(device)
        guide_trace = pyro.poutine.trace(vae.guide).get_trace(x, lengths, beta)
        # Use posterior mean for deterministic reconstruction
        mu_z = guide_trace.nodes["z"]["fn"].base_dist.loc
        mu_x, _ = vae.decode(mu_z, x.shape[1])

        for b in range(x.shape[0]):
            L = lengths[b].item()
            real_phi.extend(x[b, :L, 0].cpu().numpy())
            real_psi.extend(x[b, :L, 1].cpu().numpy())
            pred_phi.extend(mu_x[b, :L, 0].detach().cpu().numpy())
            pred_psi.extend(mu_x[b, :L, 1].detach().cpu().numpy())

    real_phi = np.degrees(real_phi)
    real_psi = np.degrees(real_psi)
    pred_phi = np.degrees(pred_phi)
    pred_psi = np.degrees(pred_psi)

    def circ_mae(a, b):
        d = np.abs(np.degrees(np.arctan2(np.sin(np.radians(a - b)),
                                          np.cos(np.radians(a - b)))))
        return float(d.mean())

    mae_phi = circ_mae(real_phi, pred_phi)
    mae_psi = circ_mae(real_psi, pred_psi)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Reconstruction Ramachandran Plot (val set, dropout OFF, posterior mean)\n"
        f"Circular MAE — φ: {mae_phi:.1f}°   ψ: {mae_psi:.1f}°",
        fontsize=11, fontweight="bold",
    )

    hex_kw = dict(gridsize=60, cmap="Blues", extent=[-180, 180, -180, 180], mincnt=1)
    ref_kw = dict(color="gray", lw=0.6, ls="--")

    hb_list = []
    for ax, phi, psi, title in [
        (axes[0], real_phi, real_psi, "Real angles"),
        (axes[1], pred_phi, pred_psi, "Reconstructed angles"),
    ]:
        hb = ax.hexbin(phi, psi, **hex_kw)
        ax.axhline(0, **ref_kw)
        ax.axvline(0, **ref_kw)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("ψ (°)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        hb_list.append((hb, ax))

    counts = np.concatenate([hb.get_array() for hb, _ in hb_list])
    counts = counts[counts > 0]
    norm = LogNorm(vmin=counts.min(), vmax=counts.max())
    for hb, ax in hb_list:
        hb.set_norm(norm)
        fig.colorbar(hb, ax=ax, label="Count (log scale)")

    plt.tight_layout()
    fig.savefig(out("diag_reconstruction_ramachandran.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 4] Ramachandran MAE — φ: {mae_phi:.1f}°  ψ: {mae_psi:.1f}°")
    return mae_phi, mae_psi


# ─── Plot 3: Generated Ramachandran ──────────────────────────────────────────

@torch.no_grad()
def plot_generated_ramachandran(vae, val_loader, device):
    """Compare real vs prior-sampled generated φ/ψ as Ramachandran density plots.

    Uses vae.generate() (z ~ N(0,I) via Pyro, VM likelihood sample) with
    sequence lengths drawn from the validation set distribution.
    """
    vae.eval()
    real_phi, real_psi = [], []
    all_lengths = []

    for x, lengths in val_loader:
        x, lengths = x.to(device), lengths.to(device)
        for b in range(x.shape[0]):
            L = lengths[b].item()
            real_phi.extend(x[b, :L, 0].cpu().numpy())
            real_psi.extend(x[b, :L, 1].cpu().numpy())
        all_lengths.extend(lengths.cpu().tolist())

    rng = np.random.default_rng(0)
    n_seqs = len(all_lengths)
    sampled_lengths = rng.choice(all_lengths, size=n_seqs, replace=True).astype(int)
    max_len = int(sampled_lengths.max())

    _, angles = vae.generate(num_samples=n_seqs, seq_len=max_len, stochastic=True)

    gen_phi, gen_psi = [], []
    for b in range(n_seqs):
        L = sampled_lengths[b]
        gen_phi.extend(angles[b, :L, 0].numpy())
        gen_psi.extend(angles[b, :L, 1].numpy())

    real_phi = np.degrees(real_phi)
    real_psi = np.degrees(real_psi)
    gen_phi  = np.degrees(gen_phi)
    gen_psi  = np.degrees(gen_psi)

    w1_phi = wasserstein_distance(real_phi, gen_phi)
    w1_psi = wasserstein_distance(real_psi, gen_psi)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Generated Ramachandran Plot  (z ~ MAF prior samples)\n"
        f"Wasserstein-1 — φ: {w1_phi:.3f}°   ψ: {w1_psi:.3f}°",
        fontsize=11, fontweight="bold",
    )

    hex_kw = dict(gridsize=60, cmap="Blues", extent=[-180, 180, -180, 180], mincnt=1)
    ref_kw = dict(color="gray", lw=0.6, ls="--")

    hb_list = []
    for ax, phi, psi, title in [
        (axes[0], real_phi, real_psi, "Real angles"),
        (axes[1], gen_phi,  gen_psi,  "Generated angles  (z ~ MAF prior)"),
    ]:
        hb = ax.hexbin(phi, psi, **hex_kw)
        ax.axhline(0, **ref_kw)
        ax.axvline(0, **ref_kw)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("ψ (°)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        hb_list.append((hb, ax))

    counts = np.concatenate([hb.get_array() for hb, _ in hb_list])
    counts = counts[counts > 0]
    norm = LogNorm(vmin=counts.min(), vmax=counts.max())
    for hb, ax in hb_list:
        hb.set_norm(norm)
        fig.colorbar(hb, ax=ax, label="Count (log scale)")

    plt.tight_layout()
    fig.savefig(out("diag_generated_ramachandran.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 5] Generated Ramachandran W1 — φ: {w1_phi:.3f}  ψ: {w1_psi:.3f}")
    return w1_phi, w1_psi


# ─── Plot 4: Kappa (concentration) distribution ──────────────────────────────

@torch.no_grad()
def plot_kappa_distribution(vae, val_loader, device, beta):
    """Histogram of decoder kappa values."""
    vae.eval()
    kappas = []

    for x, lengths in val_loader:
        x, lengths = x.to(device), lengths.to(device)
        guide_trace = pyro.poutine.trace(vae.guide).get_trace(x, lengths, beta)
        mu_z = guide_trace.nodes["z"]["fn"].base_dist.loc
        _, kappa_x = vae.decode(mu_z, x.shape[1])
        mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
        kappas.append(kappa_x[mask.unsqueeze(-1).expand_as(kappa_x)].cpu())

    kappas = torch.cat(kappas).numpy()
    angle_names = ["φ", "ψ"] if N_ANGLES == 2 else [f"angle {i}" for i in range(N_ANGLES)]

    print(f"kappa configured min: {MIN_KAPPA:.4f}")
    print(f"kappa configured max: {MAX_KAPPA:.4f}")
    print(f"kappa actual min    : {kappas.min():.4f}")
    print(f"kappa actual max    : {kappas.max():.4f}")
    print(f"kappa mean          : {kappas.mean():.4f}")
    print(f"kappa std           : {kappas.std():.4f}")
    print(f"kappa median        : {np.median(kappas):.4f}")

    fig, axes = plt.subplots(1, N_ANGLES, figsize=(6 * N_ANGLES, 4), squeeze=False)

    for i, (ax, name) in enumerate(zip(axes[0], angle_names)):
        vals = kappas[i::N_ANGLES] if kappas.ndim == 1 else kappas[:, i]
        ax.hist(vals, bins=60, color="steelblue", edgecolor="none", alpha=0.7)
        ax.axvline(vals.mean(), color="crimson", linewidth=1.2,
                   label=f"mean={vals.mean():.2f}")
        ax.axvline(MIN_KAPPA, color="orange", linewidth=1.2, linestyle="--",
                   label=f"configured min={MIN_KAPPA:.1f}")
        ax.axvline(MAX_KAPPA, color="purple", linewidth=1.2, linestyle="--",
                   label=f"configured max={MAX_KAPPA:.1f}")
        ax.set_xlabel("κ (concentration)")
        ax.set_ylabel("count")
        ax.set_title(f"Decoder κ — {name}")
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Von Mises decoder concentration (val set)\n"
        f"Configured range: [{MIN_KAPPA:.1f}, {MAX_KAPPA:.1f}]  |  "
        f"Actual range: [{kappas.min():.2f}, {kappas.max():.2f}]",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out("diag_kappa_dist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 6] configured κ: [{MIN_KAPPA:.1f}, {MAX_KAPPA:.1f}]  "
          f"actual κ: [{kappas.min():.4f}, {kappas.max():.4f}]  "
          f"mean={kappas.mean():.4f}  median={np.median(kappas):.4f}")
    return kappas


# ─── Text report ─────────────────────────────────────────────────────────────

def write_report(active, kl_mean, mae_phi, mae_psi,
                 ckpt, w1_phi_gen, w1_psi_gen, kappas=None):
    lines = []
    sep = "=" * 60

    lines += [sep, "VAE DIAGNOSTICS REPORT", sep, ""]

    lines += ["1. POSTERIOR COLLAPSE", "-" * 40]
    lines += [f"  Active units (KL>{ACTIVE_UNIT_THRESHOLD}): {active}/{LATENT_DIM}"]
    for d, kl in enumerate(kl_mean):
        flag = "✓" if kl > ACTIVE_UNIT_THRESHOLD else "✗ COLLAPSED"
        lines.append(f"    dim {d:2d}: KL = {kl:.4f} nats  {flag}")

    lines += ["", "2. RECONSTRUCTION QUALITY", "-" * 40]
    lines += [f"  Circular MAE φ : {mae_phi:.2f}°",
              f"  Circular MAE ψ : {mae_psi:.2f}°",
              f"  Random baseline: ~90° (uniform distribution)",
              f"  Perfect model  : 0°"]

    lines += ["", "3. DIAGNOSIS SUMMARY", "-" * 40]
    if active < LATENT_DIM // 2:
        lines.append("  ⚠ POSTERIOR COLLAPSE detected — most latent dims inactive.")
        lines.append("    Suggestions:")
        lines.append("    • Reduce beta_max (e.g. 0.01 instead of 0.1)")
        lines.append("    • Increase warmup_epochs (e.g. 80)")
        lines.append("    • Use free bits: replace KL with max(KL_j, λ) per dim (λ≈0.5)")
        lines.append("    • Reduce decoder capacity (smaller hidden_dim)")
    else:
        lines.append("  ✓ No collapse — majority of latent dims are active.")

    lines += ["", "4. GENERATION QUALITY", "-" * 40]
    lines += ["  Wasserstein-1 (real vs generated angle distribution, z ~ MAF prior):"]
    lines += [f"    φ: {w1_phi_gen:.3f}°",
              f"    ψ: {w1_psi_gen:.3f}°",
              "  (lower is better; ~0 = generated matches real distribution)"]
    if w1_phi_gen < 10 and w1_psi_gen < 10:
        lines.append("  ✓ Generated angles closely match the real distribution.")
    else:
        lines.append("  ⚠ Generated distribution diverges from real angles.")
        lines.append("    Suggestions:")
        lines.append("    • Check for posterior holes (see section 5)")
        lines.append("    • Increase training epochs or lower beta_max")

    lines += ["", "5. KAPPA (VON MISES CONCENTRATION)", "-" * 40]
    lines += [f"  Configured range : [{MIN_KAPPA:.4f}, {MAX_KAPPA:.4f}]"]
    if kappas is not None:
        lines += [
            f"  Actual min       : {kappas.min():.4f}",
            f"  Actual max       : {kappas.max():.4f}",
            f"  Actual mean      : {kappas.mean():.4f}",
            f"  Actual median    : {np.median(kappas):.4f}",
            f"  Actual std       : {kappas.std():.4f}",
        ]
        pct_lo = 100.0 * (kappas < MIN_KAPPA + 0.5).mean()
        pct_hi = 100.0 * (kappas > MAX_KAPPA - 0.5).mean()
        lines += [
            f"  % near configured min (< {MIN_KAPPA + 0.5:.1f}): {pct_lo:.1f}%",
            f"  % near configured max (> {MAX_KAPPA - 0.5:.1f}): {pct_hi:.1f}%",
        ]
        if pct_hi > 10:
            lines.append("  ⚠ Many kappas near the ceiling — decoder may be overconfident;"
                         " consider raising MAX_KAPPA.")
        if pct_lo > 10:
            lines.append("  ⚠ Many kappas near the floor — decoder may be underconfident;"
                         " consider lowering MIN_KAPPA.")

    lines += ["", sep, "END OF REPORT", sep]

    path = out("diag_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Report] Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nanobody VAE diagnostics")
    parser.add_argument('--run', type=str, default=None,
                        help="Experiment run name; loads from experiments/{run}/")
    args = parser.parse_args()

    global OUT_DIR
    run_name   = args.run or _latest_run()
    run_dir    = os.path.join('experiments', run_name)
    checkpoint = os.path.join(run_dir, 'vae_checkpoint.pt')
    OUT_DIR    = run_dir
    print(f"Run: {run_name}")

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae, ckpt = load_model(device, checkpoint_path=checkpoint)

    dataset      = NanobodyDataset(CSV_PATH)
    tr_idx, v_idx = dataset.create_train_val_split()
    train_loader = dataset.get_dataloader(tr_idx, batch_size=32, shuffle=False)
    val_loader   = dataset.get_dataloader(v_idx,  batch_size=32, shuffle=False)

    config_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        warmup_epochs = cfg.get('warmup_epochs', 70)
        anneal_end    = cfg.get('anneal_end', 200)
        max_beta      = cfg.get('max_beta', 0.01)
    else:
        warmup_epochs, anneal_end, max_beta = 70, 200, 0.01

    last_epoch = ckpt.get("epoch", 149)
    beta       = get_beta_schedule(last_epoch,
                                   warmup_epochs=warmup_epochs,
                                   anneal_end=anneal_end,
                                   max_beta=max_beta)
    print(f"Using β = {beta:.3f}  (epoch {last_epoch+1})\n")

    # Collect posterior parameters via Pyro guide traces
    vae.eval()
    mu_tr, lv_tr = collect_latents(vae, train_loader, device, beta)
    mu_va, lv_va = collect_latents(vae, val_loader,   device, beta)
    mu_z = torch.cat([mu_tr, mu_va])
    lv_z = torch.cat([lv_tr, lv_va])

    active, kl_mean          = plot_kl_per_dim(mu_z, lv_z)
    mae_phi, mae_psi         = plot_reconstruction_ramachandran(vae, val_loader, device, beta)
    w1_phi_gen, w1_psi_gen   = plot_generated_ramachandran(vae, val_loader, device)
    kappas = plot_kappa_distribution(vae, val_loader, device, beta)
    write_report(active, kl_mean, mae_phi, mae_psi, ckpt,
                 w1_phi_gen, w1_psi_gen, kappas=kappas)

    print(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
