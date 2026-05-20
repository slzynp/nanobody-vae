"""
run_pipeline.py
===============
End-to-end pipeline: train VAE → diagnostics → UMAP analysis.
All outputs land in experiments/{run}/ alongside the checkpoint.

Usage:
    python run_pipeline.py                  # full run, default hyperparams
    python run_pipeline.py --epochs 100     # quick smoke-test
    python run_pipeline.py --skip-umap      # skip the slow UMAP step
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

import argparse
import json
import secrets
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch

from vae import (
    NanobodyVAE, NanobodyDataset,
    N_ANGLES, HIDDEN_DIM, LATENT_DIM,
    get_beta_schedule, train_vae, plot_training_results,
)
import diagnostics
import umap_analysis


# ─── Stage 1: Training ────────────────────────────────────────────────────────

def run_training(args, run_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = NanobodyDataset()
    train_indices, val_indices = dataset.create_train_val_split()
    train_loader = dataset.get_dataloader(train_indices, batch_size=args.batch_size, shuffle=True)
    val_loader   = dataset.get_dataloader(val_indices,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}\n")

    vae = NanobodyVAE(
        input_dim=N_ANGLES, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, device=device
    ).to(device)

    train_losses, val_losses, kl_values, beta_values = train_vae(
        vae, train_loader, val_loader,
        epochs=args.epochs, device=device, patience=args.patience,
        train_indices=train_indices, val_indices=val_indices,
        warmup_epochs=args.warmup_epochs, anneal_end=args.anneal_end,
        max_beta=args.max_beta, lr=args.lr, weight_decay=args.weight_decay,
        save_dir=run_dir,
    )

    plot_training_results(
        train_losses, val_losses, kl_values, beta_values,
        save_path=os.path.join(run_dir, 'training_curves.png'),
    )

    return device


# ─── Stage 2: Diagnostics ────────────────────────────────────────────────────

def run_diagnostics(checkpoint_path, run_dir, device):
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}\n")

    diagnostics.OUT_DIR = run_dir

    vae, ckpt = diagnostics.load_model(device, checkpoint_path)

    dataset      = NanobodyDataset()
    tr_idx, v_idx = dataset.create_train_val_split()
    train_loader = dataset.get_dataloader(tr_idx, batch_size=32, shuffle=False)
    val_loader   = dataset.get_dataloader(v_idx,  batch_size=32, shuffle=False)

    last_epoch = ckpt.get('epoch', 149)
    beta       = get_beta_schedule(last_epoch)
    print(f"Using β = {beta:.3f}  (epoch {last_epoch + 1})\n")

    vae.eval()
    mu_tr, lv_tr, _ = diagnostics.collect_latents(vae, train_loader, device)
    mu_va, lv_va, _ = diagnostics.collect_latents(vae, val_loader,   device)
    mu_z = torch.cat([mu_tr, mu_va])
    lv_z = torch.cat([lv_tr, lv_va])

    _, _, kappa_vals = diagnostics.compute_loss_eval_mode(vae, val_loader, device, beta)

    active, kl_mean  = diagnostics.plot_kl_per_dim(mu_z, lv_z)
    diagnostics.plot_kappa(kappa_vals)
    loss_results     = diagnostics.plot_loss_modes(vae, train_loader, val_loader, device, beta)
    mae_phi, mae_psi = diagnostics.plot_reconstruction(vae, val_loader, device)
    traversal_range  = diagnostics.plot_latent_traversal(vae, device)
    diagnostics.write_report(active, kl_mean, kappa_vals, loss_results,
                             mae_phi, mae_psi, traversal_range, ckpt)

    print(f"\nDiagnostics outputs → {run_dir}/")


# ─── Stage 3: UMAP analysis ──────────────────────────────────────────────────

def run_umap(checkpoint_path, run_dir):
    print(f"\n{'='*70}")
    print("UMAP ANALYSIS")
    print(f"{'='*70}\n")

    device = umap_analysis.DEVICE
    np.random.seed(umap_analysis.RANDOM_SEED)
    torch.manual_seed(umap_analysis.RANDOM_SEED)

    vae, train_indices, val_indices = umap_analysis.load_model(checkpoint_path, device)

    df = pd.read_csv(umap_analysis.DATA_FILE)
    summary = (
        pd.read_csv("nanobody_summary.tsv", sep="\t", usecols=["pdb", "short_header"])
          .drop_duplicates("pdb")
          .rename(columns={"pdb": "pdb_id"})
    )
    df = df.merge(summary, on="pdb_id", how="left")
    df["short_header"] = df["short_header"].fillna("unknown")

    dataset = NanobodyDataset(file=umap_analysis.DATA_FILE)
    is_train, is_val, split_labels = umap_analysis.build_split_mask(
        len(dataset), train_indices, val_indices
    )

    mu_z, seq_lengths = umap_analysis.encode_dataset(vae, dataset, device)
    features          = umap_analysis.extract_features(df)

    print("\nRunning UMAP (fit on train, transform all)...")
    embedding = umap_analysis.run_umap(mu_z, is_train)

    umap_analysis.plot_umap(embedding, features, df, seq_lengths, split_labels,
                            save_path=os.path.join(run_dir, 'umap_colorings.png'))
    umap_analysis.plot_dim_correlations(mu_z, features,
                                        save_path=os.path.join(run_dir, 'dim_correlations.png'))
    umap_analysis.plot_top_dims(mu_z, features,
                                save_path=os.path.join(run_dir, 'top_dims_cdr3.png'))

    recon_loss = umap_analysis.compute_per_sequence_loss(vae, dataset, device)
    outlier_indices, isolation_scores = umap_analysis.find_outliers(embedding)

    umap_analysis.report_outliers(df, outlier_indices, isolation_scores, recon_loss,
                                   embedding, is_train, train_indices)
    umap_analysis.plot_outlier_spike(embedding, recon_loss, df, outlier_indices,
                                     save_path=os.path.join(run_dir, 'outlier_spike_analysis.png'))

    print(f"\nUMAP outputs → {run_dir}/")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end VAE pipeline")
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--warmup-epochs', type=int,   default=60)
    parser.add_argument('--anneal-end',    type=int,   default=170)
    parser.add_argument('--max-beta',      type=float, default=0.01)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--weight-decay',  type=float, default=1e-5)
    parser.add_argument('--skip-diag',     action='store_true', help="Skip diagnostics stage")
    parser.add_argument('--skip-umap',     action='store_true', help="Skip UMAP stage")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    timestamp  = datetime.now().strftime('%Y%m%d')
    run_name   = f"run_{timestamp}_{secrets.token_hex(3)}"
    run_dir    = os.path.join('experiments', run_name)
    checkpoint = os.path.join(run_dir, 'vae_checkpoint.pt')
    os.makedirs(run_dir, exist_ok=True)

    config = {**vars(args), 'run_name': run_name}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Run: {run_name}")
    print(f"All outputs → {run_dir}/\n")

    device = run_training(args, run_dir)

    if not args.skip_diag:
        run_diagnostics(checkpoint, run_dir, device)

    if not args.skip_umap:
        run_umap(checkpoint, run_dir)

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE  →  {run_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
