"""
VAE Diagnostics
===============
Checks two known symptoms:
  1. Posterior collapse  — KL ≈ 0, latent dims not used by encoder
  2. Val < Train loss    — overfitting or train/eval mode discrepancy

Run after training:  python diagnostics.py

Outputs:
  diag_kl_per_dim.png          per-dimension KL and active-unit count
  diag_kappa_dist.png          decoder concentration (κ) distribution
  diag_loss_modes.png          train vs val loss in train-mode vs eval-mode
  diag_reconstruction.png      real vs reconstructed φ/ψ on val set
  diag_latent_traversal.png    vary each latent dim ± 3σ, show angle response
  diag_report.txt              numerical summary
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vae import (
    NanobodyVAE, NanobodyDataset,
    N_ANGLES, HIDDEN_DIM, LATENT_DIM, MAX_LEN,
    collate_fn, get_beta_schedule,
)

CSV_PATH    = "nanobodies_filtered.csv"
OUT_DIR     = None   # set in main() from the run directory
ACTIVE_UNIT_THRESHOLD = 0.1   # nats; standard in the literature

def out(name):
    return os.path.join(OUT_DIR, name)


def _latest_run():
    exp_dir = 'experiments'
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError("No 'experiments/' directory. Run vae.py first.")
    runs = sorted(
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
        and os.path.exists(os.path.join(exp_dir, d, 'vae_checkpoint.pt'))
    )
    if not runs:
        raise FileNotFoundError("No checkpoint found in experiments/. Run vae.py first.")
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
def collect_latents(vae, loader, device):
    """Return mu_z, log_var_z, seq_lengths for every sample."""
    all_mu, all_lv, all_lens = [], [], []
    for x, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        mu_z, lv = vae.encode(x, lengths)
        all_mu.append(mu_z.cpu())
        all_lv.append(lv.cpu())
        all_lens.append(lengths.cpu())
    return (torch.cat(all_mu),
            torch.cat(all_lv),
            torch.cat(all_lens))


@torch.no_grad()
def compute_loss_eval_mode(vae, loader, device, beta):
    """Reconstruction MAE and KL with dropout OFF (eval mode)."""
    vae.eval()
    recon_errors, kl_vals, kappa_vals = [], [], []

    for x, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        mu_z, lv   = vae.encode(x, lengths)
        z          = mu_z                              # use posterior mean (no noise)
        mu_x, kappa_x = vae.decode(z, x.shape[1])

        mask = (torch.arange(x.shape[1], device=device)[None, :]
                < lengths[:, None]).float().unsqueeze(-1)

        # Circular MAE (degrees)
        diff = torch.atan2(torch.sin(x - mu_x), torch.cos(x - mu_x))
        mae  = (diff.abs() * mask).sum(dim=[1, 2]) / lengths.float()
        recon_errors.append(np.degrees(mae.cpu().numpy()))

        # KL per sample
        kl = -0.5 * (1 + lv - mu_z.pow(2) - lv.exp()).sum(-1)
        kl_vals.append(kl.cpu().numpy())

        # kappa values at non-padded positions (flatten phi+psi into 1D)
        kappa_flat = kappa_x[mask.squeeze(-1).bool()].flatten()
        kappa_vals.append(kappa_flat.cpu().numpy())

    return (np.concatenate(recon_errors),
            np.concatenate(kl_vals),
            np.concatenate(kappa_vals))


@torch.no_grad()
def compute_elbo_loss(vae, loader, device, beta, train_mode):
    """Negative ELBO (free-bits, per-residue) in train or eval mode."""
    if train_mode:
        vae.train()
    else:
        vae.eval()

    losses = []
    for x, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        losses.append(vae.elbo_loss(x, lengths, beta).item())

    return np.mean(losses)


# ─── Plot 1: KL per latent dimension ─────────────────────────────────────────

def plot_kl_per_dim(mu_z, lv_z):
    # Per-dim KL: 0.5*(mu²+sigma²−1−log sigma²), averaged over samples
    kl_per_dim = 0.5 * (mu_z.pow(2) + lv_z.exp() - 1 - lv_z)
    kl_mean    = kl_per_dim.mean(0).numpy()       # [LATENT_DIM]
    kl_std     = kl_per_dim.std(0).numpy()

    # Variance of mu_z per dim (proxy for "how much the encoder uses each dim")
    mu_var = mu_z.var(0).numpy()

    active = (kl_mean > ACTIVE_UNIT_THRESHOLD).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Posterior Collapse Check  —  Active units: {active}/{LATENT_DIM}"
                 f"  (threshold {ACTIVE_UNIT_THRESHOLD} nats)",
                 fontsize=12, fontweight="bold")

    dims = np.arange(LATENT_DIM)

    # KL bar chart
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

    # Variance of mu_z
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


# ─── Plot 2: Kappa distribution ───────────────────────────────────────────────

def plot_kappa(kappa_vals):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Decoder Concentration (κ) — Low κ everywhere = high uncertainty = collapse",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.hist(kappa_vals, bins=60, color="#8e44ad", alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(1.0, color="red", lw=1.5, ls="--", label="κ = 1")
    ax.set_xlabel("κ (concentration)")
    ax.set_ylabel("Count")
    ax.set_title("κ distribution (linear scale)")
    ax.legend(fontsize=8)
    ax.text(0.65, 0.92, f"Median κ: {np.median(kappa_vals):.3f}",
            transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    ax.hist(np.log10(kappa_vals + 1e-6), bins=60, color="#8e44ad", alpha=0.85,
            edgecolor="white", lw=0.3)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="κ = 1")
    ax.set_xlabel("log₁₀(κ)")
    ax.set_ylabel("Count")
    ax.set_title("κ distribution (log scale)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out("diag_kappa_dist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 2] Median κ = {np.median(kappa_vals):.4f}  "
          f"(κ<0.1: {(kappa_vals<0.1).mean()*100:.1f}%)")


# ─── Plot 3: Train vs Val — train-mode vs eval-mode ──────────────────────────

def plot_loss_modes(vae, train_loader, val_loader, device, beta):
    print("[Diag 3] Computing train/val losses in train and eval mode (takes a moment)...")
    results = {}
    for split, loader in [("Train", train_loader), ("Val", val_loader)]:
        for mode_label, train_mode in [("train_mode", True),
                                        ("eval_mode ", False)]:
            loss = compute_elbo_loss(vae, loader, device, beta, train_mode)
            results[f"{split} / {mode_label}"] = loss
            print(f"  {split:5s} {mode_label}: {loss:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(results.keys())
    values = list(results.values())
    colors = ["#3498db", "#85c1e9", "#e74c3c", "#f1948a"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Negative ELBO (per residue, per sample)")
    ax.set_title("Loss breakdown: train vs val in train mode and eval mode",
                 fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    fig.savefig(out("diag_loss_modes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return results


# ─── Plot 4: Reconstruction quality ──────────────────────────────────────────

@torch.no_grad()
def plot_reconstruction(vae, val_loader, device):
    vae.eval()
    real_phi, real_psi, pred_phi, pred_psi = [], [], [], []

    for x, lengths in val_loader:
        x, lengths = x.to(device), lengths.to(device)
        mu_z, lv   = vae.encode(x, lengths)
        mu_x, _    = vae.decode(mu_z, x.shape[1])

        mask = (torch.arange(x.shape[1], device=device)[None, :]
                < lengths[:, None])                                    # (B, T)
        for b in range(x.shape[0]):
            L = lengths[b].item()
            real_phi.extend(x[b, :L, 0].cpu().numpy())
            real_psi.extend(x[b, :L, 1].cpu().numpy())
            pred_phi.extend(mu_x[b, :L, 0].cpu().numpy())
            pred_psi.extend(mu_x[b, :L, 1].cpu().numpy())

    real_phi = np.degrees(real_phi); real_psi = np.degrees(real_psi)
    pred_phi = np.degrees(pred_phi); pred_psi = np.degrees(pred_psi)

    # Circular MAE
    def circ_mae(a, b):
        d = np.abs(np.degrees(np.arctan2(np.sin(np.radians(a-b)),
                                          np.cos(np.radians(a-b)))))
        return d.mean()

    mae_phi = circ_mae(real_phi, pred_phi)
    mae_psi = circ_mae(real_psi, pred_psi)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Reconstruction Quality (val set, dropout OFF)\n"
                 f"Circular MAE — φ: {mae_phi:.1f}°  ψ: {mae_psi:.1f}°",
                 fontsize=11, fontweight="bold")

    for ax, r, p, name in [(axes[0], real_phi, pred_phi, "φ"),
                            (axes[1], real_psi, pred_psi, "ψ")]:
        ax.scatter(r, p, s=2, alpha=0.25, c="#2c3e50", rasterized=True)
        lo, hi = -180, 180
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Real {name} (°)"); ax.set_ylabel(f"Predicted {name} (°)")
        ax.set_title(f"{name} — MAE {circ_mae(r, p):.1f}°")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out("diag_reconstruction.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 4] Reconstruction MAE — φ: {mae_phi:.1f}°  ψ: {mae_psi:.1f}°")
    return mae_phi, mae_psi


# ─── Plot 5: Latent traversal ─────────────────────────────────────────────────

@torch.no_grad()
def plot_latent_traversal(vae, device, n_steps=7, sigma=3.0):
    """Vary each latent dim from −σ to +σ; plot mean φ across all positions."""
    vae.eval()
    n_dims = LATENT_DIM
    values = np.linspace(-sigma, sigma, n_steps)

    # Mean φ and ψ (radians) for each (dim, value) pair
    phi_grid = np.zeros((n_dims, n_steps))
    psi_grid = np.zeros((n_dims, n_steps))

    for d in range(n_dims):
        for v_i, v in enumerate(values):
            z = torch.zeros(1, n_dims, device=device)
            z[0, d] = v
            mu_x, _ = vae.decode(z, MAX_LEN)           # (1, MAX_LEN, 2)
            phi_grid[d, v_i] = np.degrees(mu_x[0, :, 0].mean().item())
            psi_grid[d, v_i] = np.degrees(mu_x[0, :, 1].mean().item())

    # Range of response for each dim
    phi_range = phi_grid.max(axis=1) - phi_grid.min(axis=1)
    psi_range = psi_grid.max(axis=1) - psi_grid.min(axis=1)
    total_range = phi_range + psi_range

    ncols = 4
    nrows = (n_dims + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    fig.suptitle(f"Latent Traversal (z_d ∈ [−{sigma}σ, +{sigma}σ], all other dims = 0)\n"
                 "Flat lines → dim not used by decoder (collapse)",
                 fontsize=11, fontweight="bold")
    axes = axes.flatten()

    for d in range(n_dims):
        ax = axes[d]
        ax.plot(values, phi_grid[d], "b-o", ms=4, label="mean φ")
        ax.plot(values, psi_grid[d], "r-s", ms=4, label="mean ψ")
        ax.set_title(f"Dim {d}  (Δ={total_range[d]:.1f}°)", fontsize=9,
                     fontweight="bold" if total_range[d] > 10 else "normal")
        ax.set_xlabel(f"z_{d}", fontsize=7)
        ax.set_ylabel("Mean angle (°)", fontsize=7)
        ax.set_ylim(-180, 180)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        if d == 0:
            ax.legend(fontsize=7)

    for d in range(n_dims, len(axes)):
        axes[d].set_visible(False)

    plt.tight_layout()
    fig.savefig(out("diag_latent_traversal.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 5] Latent traversal — "
          f"dims with Δ>10°: {(total_range>10).sum()}/{n_dims}")
    return total_range


# ─── Text report ─────────────────────────────────────────────────────────────

def write_report(active, kl_mean, kappa_vals, loss_results, mae_phi, mae_psi,
                 traversal_range, ckpt):
    lines = []
    sep = "=" * 60

    lines += [sep, "VAE DIAGNOSTICS REPORT", sep, ""]

    lines += ["1. POSTERIOR COLLAPSE", "-" * 40]
    lines += [f"  Active units (KL>{ACTIVE_UNIT_THRESHOLD}): {active}/{LATENT_DIM}"]
    for d, kl in enumerate(kl_mean):
        flag = "✓" if kl > ACTIVE_UNIT_THRESHOLD else "✗ COLLAPSED"
        lines.append(f"    dim {d:2d}: KL = {kl:.4f} nats  {flag}")

    lines += ["", "  Decoder response to latent traversal (±3σ):"]
    for d, r in enumerate(traversal_range):
        flag = "✓ active" if r > 10 else "✗ flat"
        lines.append(f"    dim {d:2d}: Δangle = {r:.1f}°  {flag}")

    lines += ["", "2. KAPPA (DECODER CONFIDENCE)", "-" * 40]
    lines += [f"  Median κ  : {np.median(kappa_vals):.4f}",
              f"  Mean κ    : {np.mean(kappa_vals):.4f}",
              f"  κ < 0.1   : {(kappa_vals<0.1).mean()*100:.1f}%  "
              f"(high → decoder uncertain → collapse sign)",
              f"  κ < 1.0   : {(kappa_vals<1.0).mean()*100:.1f}%"]

    lines += ["", "3. TRAIN vs VAL LOSS BREAKDOWN", "-" * 40]
    for label, val in loss_results.items():
        lines.append(f"  {label}: {val:.4f}")

    lines += ["", "4. RECONSTRUCTION QUALITY", "-" * 40]
    lines += [f"  Circular MAE φ : {mae_phi:.2f}°",
              f"  Circular MAE ψ : {mae_psi:.2f}°",
              f"  Random baseline: ~90° (uniform distribution)",
              f"  Perfect model  : 0°"]

    lines += ["", "5. DIAGNOSIS SUMMARY", "-" * 40]
    if active < LATENT_DIM // 2:
        lines.append("  ⚠ POSTERIOR COLLAPSE detected — most latent dims inactive.")
        lines.append("    Suggestions:")
        lines.append("    • Reduce beta_max (e.g. 0.01 instead of 0.1)")
        lines.append("    • Increase warmup_epochs (e.g. 80)")
        lines.append("    • Use free bits: replace KL with max(KL_j, λ) per dim (λ≈0.5)")
        lines.append("    • Reduce decoder capacity (smaller hidden_dim)")
    else:
        lines.append("  ✓ No collapse — majority of latent dims are active.")

    tv_eval_gap = loss_results["Val / eval_mode "] \
                - loss_results["Train / eval_mode "]
    if abs(tv_eval_gap) < 0.05 * abs(loss_results["Train / eval_mode "]):
        lines.append("  ✓ Train ≈ Val in eval mode → no overfitting.")
    elif tv_eval_gap < -0.05:
        lines.append("  ⚠ Val < Train in eval mode — possible data leakage or")
        lines.append("    val set is easier than train set.")

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

    # Build same train/val split as training
    dataset      = NanobodyDataset(CSV_PATH)
    tr_idx, v_idx = dataset.create_train_val_split()
    train_loader = dataset.get_dataloader(tr_idx, batch_size=32, shuffle=False)
    val_loader   = dataset.get_dataloader(v_idx,  batch_size=32, shuffle=False)

    # Use the final beta for loss mode comparison
    last_epoch = ckpt.get("epoch", 149)
    beta       = get_beta_schedule(last_epoch)
    print(f"Using β = {beta:.3f}  (epoch {last_epoch+1})\n")

    # ── Collect latent stats (eval mode, no dropout)
    vae.eval()
    mu_tr, lv_tr, _ = collect_latents(vae, train_loader, device)
    mu_va, lv_va, _ = collect_latents(vae, val_loader,   device)
    mu_z = torch.cat([mu_tr, mu_va])
    lv_z = torch.cat([lv_tr, lv_va])

    # ── Per-dim kappa
    _, _, kappa_vals = compute_loss_eval_mode(vae, val_loader, device, beta)

    active, kl_mean   = plot_kl_per_dim(mu_z, lv_z)
    plot_kappa(kappa_vals)
    loss_results      = plot_loss_modes(vae, train_loader, val_loader, device, beta)
    mae_phi, mae_psi  = plot_reconstruction(vae, val_loader, device)
    traversal_range   = plot_latent_traversal(vae, device)
    write_report(active, kl_mean, kappa_vals, loss_results,
                 mae_phi, mae_psi, traversal_range, ckpt)

    print(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
