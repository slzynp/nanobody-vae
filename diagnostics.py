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
from pyro.infer import TraceMeanField_ELBO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, norm as scipy_norm

from vae_svi import (
    NanobodyVAE, NanobodyDataset,
    N_ANGLES, HIDDEN_DIM, LATENT_DIM,
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
    """Negative ELBO (TraceMeanField) in train or eval mode."""
    elbo = TraceMeanField_ELBO()
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


# ─── Plot 2: Aggregate posterior vs N(0,1) ───────────────────────────────────

POSTERIOR_HOLE_THRESHOLD = 0.3

def plot_aggregate_posterior(mu_z, save_path):
    """Diagnose posterior holes by comparing the aggregate posterior to N(0,1)."""
    mu_np = mu_z.numpy()
    N, D  = mu_np.shape

    rng         = np.random.default_rng(0)
    ref_samples = rng.standard_normal(max(N, 10_000))
    wasserstein_dists = np.array([
        wasserstein_distance(mu_np[:, d], ref_samples)
        for d in range(D)
    ])

    print("[Diag 2] Wasserstein-1 distance to N(0,1) per dimension:")
    for d, w in enumerate(wasserstein_dists):
        flag = "  ← HOLE" if w > POSTERIOR_HOLE_THRESHOLD else ""
        print(f"  dim {d:2d}: W1 = {w:.4f}{flag}")

    top5_idx = np.argsort(wasserstein_dists)[::-1][:5]
    print("[Diag 2] Top 5 most deviant dimensions:")
    for rank, d in enumerate(top5_idx):
        print(f"  #{rank+1}: dim {d:2d}  W1={wasserstein_dists[d]:.4f}"
              f"  mean={mu_np[:, d].mean():.3f}  std={mu_np[:, d].std():.3f}")

    nrows_grid, ncols_grid = 4, 8
    fig = plt.figure(figsize=(ncols_grid * 2.5, nrows_grid * 2.2 + 5))
    gs  = fig.add_gridspec(
        nrows_grid + 1, ncols_grid,
        height_ratios=[2.2] * nrows_grid + [4.5],
        hspace=0.55, wspace=0.35,
    )

    x_ref = np.linspace(-4, 4, 200)
    y_ref = scipy_norm.pdf(x_ref)

    for d in range(D):
        ax = fig.add_subplot(gs[d // ncols_grid, d % ncols_grid])
        ax.hist(mu_np[:, d], bins=30, density=True, color="#3498db",
                alpha=0.70, edgecolor="white", linewidth=0.3)
        ax.plot(x_ref, y_ref, "r-", lw=1.2)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        is_hole = wasserstein_dists[d] > POSTERIOR_HOLE_THRESHOLD
        ax.set_title(f"z{d}  W={wasserstein_dists[d]:.2f}",
                     fontsize=7,
                     fontweight="bold" if is_hole else "normal",
                     color="red"       if is_hole else "black")
        ax.set_xticks([-3, 0, 3])
        ax.tick_params(labelsize=6)
        ax.set_yticks([])

    ax_s  = fig.add_subplot(gs[nrows_grid, :])
    dims  = np.arange(D)
    width = 0.35
    ax_s.bar(dims - width / 2, mu_np.mean(0), width,
             label="mean(μ_z)", color="#e74c3c", alpha=0.80)
    ax_s.bar(dims + width / 2, mu_np.std(0),  width,
             label="std(μ_z)",  color="#2ecc71", alpha=0.80)
    ax_s.axhline(0, color="red",   lw=1.0, ls="--", alpha=0.5)
    ax_s.axhline(1, color="green", lw=1.5, ls="--", label="std = 1 (target)")
    ax_s.set_xlabel("Latent dimension")
    ax_s.set_ylabel("Value")
    ax_s.set_title(
        "Per-dimension mean and std of μ_z  (healthy: mean≈0, std≈1)",
        fontsize=10, fontweight="bold",
    )
    ax_s.legend(fontsize=8)
    ax_s.set_xticks(dims)
    ax_s.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        "Aggregate Posterior vs N(0,1) Prior — Posterior Hole Diagnosis\n"
        "(red title = W1 > 0.3, red curve = N(0,1) reference)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 2] Aggregate posterior plot saved → {save_path}")
    return wasserstein_dists


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
    ax.set_ylabel("Negative ELBO (TraceMeanField, per sample)")
    ax.set_title("Loss breakdown: train vs val in train mode and eval mode",
                 fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    fig.savefig(out("diag_loss_modes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return results


# ─── Plot 4: Reconstruction Ramachandran ─────────────────────────────────────

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

    for ax, phi, psi, title in [
        (axes[0], real_phi, real_psi, "Real angles"),
        (axes[1], pred_phi, pred_psi, "Reconstructed angles"),
    ]:
        hb = ax.hexbin(phi, psi, **hex_kw)
        fig.colorbar(hb, ax=ax, label="Count")
        ax.axhline(0, **ref_kw)
        ax.axvline(0, **ref_kw)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("ψ (°)")
        ax.set_title(title, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out("diag_reconstruction_ramachandran.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 4] Ramachandran MAE — φ: {mae_phi:.1f}°  ψ: {mae_psi:.1f}°")
    return mae_phi, mae_psi


# ─── Plot 5: Generated Ramachandran ──────────────────────────────────────────

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
        f"Generated Ramachandran Plot  (z ~ N(0,I), prior samples)\n"
        f"Wasserstein-1 — φ: {w1_phi:.3f}°   ψ: {w1_psi:.3f}°",
        fontsize=11, fontweight="bold",
    )

    hex_kw = dict(gridsize=60, cmap="Blues", extent=[-180, 180, -180, 180], mincnt=1)
    ref_kw = dict(color="gray", lw=0.6, ls="--")

    for ax, phi, psi, title in [
        (axes[0], real_phi, real_psi, "Real angles"),
        (axes[1], gen_phi,  gen_psi,  "Generated angles  (z ~ N(0,I))"),
    ]:
        hb = ax.hexbin(phi, psi, **hex_kw)
        fig.colorbar(hb, ax=ax, label="Count")
        ax.axhline(0, **ref_kw)
        ax.axvline(0, **ref_kw)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("ψ (°)")
        ax.set_title(title, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out("diag_generated_ramachandran.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag 5] Generated Ramachandran W1 — φ: {w1_phi:.3f}  ψ: {w1_psi:.3f}")
    return w1_phi, w1_psi


# ─── Text report ─────────────────────────────────────────────────────────────

def write_report(active, kl_mean, loss_results, mae_phi, mae_psi,
                 ckpt, wasserstein_dists, w1_phi_gen, w1_psi_gen):
    lines = []
    sep = "=" * 60

    lines += [sep, "VAE DIAGNOSTICS REPORT", sep, ""]

    lines += ["1. POSTERIOR COLLAPSE", "-" * 40]
    lines += [f"  Active units (KL>{ACTIVE_UNIT_THRESHOLD}): {active}/{LATENT_DIM}"]
    for d, kl in enumerate(kl_mean):
        flag = "✓" if kl > ACTIVE_UNIT_THRESHOLD else "✗ COLLAPSED"
        lines.append(f"    dim {d:2d}: KL = {kl:.4f} nats  {flag}")

    lines += ["", "2. TRAIN vs VAL LOSS BREAKDOWN", "-" * 40]
    for label, val in loss_results.items():
        lines.append(f"  {label}: {val:.4f}")

    lines += ["", "3. RECONSTRUCTION QUALITY", "-" * 40]
    lines += [f"  Circular MAE φ : {mae_phi:.2f}°",
              f"  Circular MAE ψ : {mae_psi:.2f}°",
              f"  Random baseline: ~90° (uniform distribution)",
              f"  Perfect model  : 0°"]

    lines += ["", "4. DIAGNOSIS SUMMARY", "-" * 40]
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

    lines += ["", "5. AGGREGATE POSTERIOR", "-" * 40]
    n_holes = (wasserstein_dists > POSTERIOR_HOLE_THRESHOLD).sum()
    lines += [f"  Wasserstein-1 distance to N(0,1) per dimension "
              f"(threshold = {POSTERIOR_HOLE_THRESHOLD}):"]
    for d, w in enumerate(wasserstein_dists):
        flag = "  ← POTENTIAL HOLE" if w > POSTERIOR_HOLE_THRESHOLD else ""
        lines.append(f"    dim {d:2d}: W1 = {w:.4f}{flag}")

    lines += ["", f"  Dimensions flagged as potential holes: {n_holes}/{LATENT_DIM}"]

    top5_idx = np.argsort(wasserstein_dists)[::-1][:5]
    lines += ["", "  Top 5 most deviant dimensions:"]
    for rank, d in enumerate(top5_idx):
        lines.append(f"    #{rank+1}: dim {d:2d}  W1 = {wasserstein_dists[d]:.4f}")

    if n_holes > 0:
        lines += ["",
                  "  ⚠ Posterior hole(s) detected — the aggregate posterior is not",
                  "    covering the prior. Encoder is placing mass in a sub-region,",
                  "    leaving most of the prior empty (holes).",
                  "    Suggestions:",
                  "    • Increase beta or slow down KL annealing to spread the posterior",
                  "    • Add an MMD or aggregate posterior regulariser",
                  "    • Inspect diag_aggregate_posterior.png for affected dims"]
    else:
        lines.append("  ✓ No posterior holes detected — aggregate posterior ≈ N(0,1).")

    lines += ["", "6. GENERATION QUALITY", "-" * 40]
    lines += ["  Wasserstein-1 (real vs generated angle distribution):"]
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

    last_epoch = ckpt.get("epoch", 149)
    beta       = get_beta_schedule(last_epoch)
    print(f"Using β = {beta:.3f}  (epoch {last_epoch+1})\n")

    # Collect posterior parameters via Pyro guide traces
    vae.eval()
    mu_tr, lv_tr = collect_latents(vae, train_loader, device, beta)
    mu_va, lv_va = collect_latents(vae, val_loader,   device, beta)
    mu_z = torch.cat([mu_tr, mu_va])
    lv_z = torch.cat([lv_tr, lv_va])

    active, kl_mean   = plot_kl_per_dim(mu_z, lv_z)
    wasserstein_dists = plot_aggregate_posterior(
        mu_z, os.path.join(OUT_DIR, "diag_aggregate_posterior.png")
    )
    loss_results          = plot_loss_modes(vae, train_loader, val_loader, device, beta)
    mae_phi, mae_psi      = plot_reconstruction_ramachandran(vae, val_loader, device, beta)
    w1_phi_gen, w1_psi_gen = plot_generated_ramachandran(vae, val_loader, device)
    write_report(active, kl_mean, loss_results, mae_phi, mae_psi, ckpt, wasserstein_dists,
                 w1_phi_gen, w1_psi_gen)

    print(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
