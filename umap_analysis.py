import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'  # avoid OpenMP conflict with PyTorch on macOS

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

import argparse
import importlib.util
if importlib.util.find_spec("umap") is None:
    raise ImportError("Run: pip install umap-learn")

from vae import NanobodyVAE, NanobodyDataset, collate_fn, MAX_LEN, N_ANGLES, HIDDEN_DIM, LATENT_DIM

# ============================================================================
# CONFIG
# ============================================================================
DATA_FILE   = "nanobodies_filtered.csv"
DEVICE      = torch.device("cpu")
RANDOM_SEED = 42
N_NEIGHBORS = 20
MIN_DIST    = 0.01
SPIKE_EPOCH = 280
BATCH_SIZE  = 32
N_OUTLIERS  = 5


# ============================================================================
# HELPERS
# ============================================================================

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


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(checkpoint_path, device):
    vae = NanobodyVAE(
        input_dim=N_ANGLES, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, device=device
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")

    train_indices = ckpt.get('train_indices')
    val_indices   = ckpt.get('val_indices')
    if train_indices is None or val_indices is None:
        raise RuntimeError("Checkpoint has no train/val split — retrain with the updated vae.py")
    print(f"Split loaded: {len(train_indices)} train, {len(val_indices)} val")
    return vae, train_indices, val_indices


# ============================================================================
# ENCODE
# ============================================================================

def encode_dataset(vae, dataset, device, batch_size=64):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    all_mu, all_lengths = [], []

    with torch.no_grad():
        for x, seq_lengths in loader:
            x           = x.to(device)
            seq_lengths = seq_lengths.to(device)
            mu_z, _     = vae.encode(x, seq_lengths)
            all_mu.append(mu_z.cpu())
            all_lengths.append(seq_lengths.cpu())

    mu_z    = torch.cat(all_mu,     dim=0).numpy()
    lengths = torch.cat(all_lengths,dim=0).numpy()
    print(f"Encoded {len(mu_z)} sequences, latent dim = {mu_z.shape[1]}")
    return mu_z, lengths


def build_split_mask(n_total, train_indices, val_indices):
    """Return boolean arrays and a string label array aligned to the full dataset."""
    is_train = np.zeros(n_total, dtype=bool)
    is_train[train_indices] = True
    is_val   = np.zeros(n_total, dtype=bool)
    is_val[val_indices]     = True
    split_labels = np.where(is_train, 'train', 'val')
    return is_train, is_val, split_labels


# ============================================================================
# EXTRACT CDR3 FEATURES DIRECTLY FROM METADATA
# Uses pre-computed cdr3_torsions column — no boundary guessing needed.
# ============================================================================

def extract_features(df):
    cdr3_len, mean_phi, mean_psi, var_phi, var_psi = [], [], [], [], []

    for _, row in df.iterrows():
        t   = np.deg2rad(np.array(eval(row['cdr3_torsions']), dtype=np.float32))
        phi = t[:, 0]
        psi = t[:, 1]
        n   = len(phi)

        cdr3_len.append(n)
        mean_phi.append(np.mean(phi) if n > 0 else 0.0)
        mean_psi.append(np.mean(psi) if n > 0 else 0.0)
        var_phi.append(np.var(phi)   if n > 1 else 0.0)
        var_psi.append(np.var(psi)   if n > 1 else 0.0)

    features = {
        "cdr3_length"   : np.array(cdr3_len),
        "cdr3_mean_phi" : np.array(mean_phi),
        "cdr3_mean_psi" : np.array(mean_psi),
        "cdr3_var_phi"  : np.array(var_phi),
        "cdr3_var_psi"  : np.array(var_psi),
    }

    cl = features['cdr3_length']
    print(f"\nCDR3 length: mean={cl.mean():.1f}  std={cl.std():.1f}  "
          f"min={cl.min():.0f}  max={cl.max():.0f}")
    return features


# ============================================================================
# CATEGORICAL COLOR HELPER
# ============================================================================

def categorical_colors(labels):
    unique  = sorted(set(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    codes   = np.array([mapping[l] for l in labels])
    return codes, unique


# ============================================================================
# RUN UMAP
# ============================================================================

def run_umap(mu_z, is_train):
    """Fit UMAP on training points only, then transform the full dataset.

    Runs in a subprocess to avoid the macOS OpenMP segfault that occurs when
    PyTorch and numba both try to load libomp in the same process.
    """
    import subprocess
    import sys
    import textwrap
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path    = os.path.join(tmpdir, "input.npz")
        out_path    = os.path.join(tmpdir, "embedding.npy")
        script_path = os.path.join(tmpdir, "run_umap.py")

        np.savez(inp_path, mu_z=mu_z, is_train=is_train)

        script = textwrap.dedent(f"""\
            import os
            os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['NUMBA_NUM_THREADS'] = '1'
            import numpy as np
            import umap

            data     = np.load({inp_path!r})
            mu_z     = data['mu_z']
            is_train = data['is_train'].astype(bool)

            reducer = umap.UMAP(
                n_components=2, n_neighbors={N_NEIGHBORS},
                min_dist={MIN_DIST}, metric='euclidean',
                random_state={RANDOM_SEED}, verbose=True, n_jobs=1
            )
            reducer.fit(mu_z[is_train])
            emb = reducer.transform(mu_z)
            np.save({out_path!r}, emb)
            print(f'UMAP embedding: {{emb.shape}}  (fit on {{is_train.sum()}} train points)')
        """)

        with open(script_path, "w") as f:
            f.write(script)

        subprocess.run([sys.executable, script_path], check=True)
        emb = np.load(out_path)

    return emb


# ============================================================================
# PLOT 1 — UMAP colored by CDR3 features and metadata
# ============================================================================

def plot_umap(embedding, features, df, seq_lengths, split_labels,
              save_path="umap_colorings.png"):
    cmap_cat = plt.get_cmap("tab10")

    # Simplify species to top 4 + other
    top_sp  = df['species'].value_counts().index[:4].tolist()
    species = df['species'].apply(lambda s: s if s in top_sp else "other")
    sp_codes, sp_labels = categorical_colors(species)

    sc_codes, sc_labels = categorical_colors(df['heavy_subclass'].fillna('unknown'))

    antigen = df['antigen_type'].apply(
        lambda s: s.split('|')[0].strip() if isinstance(s, str) else 'unknown'
    )
    ag_codes, ag_labels = categorical_colors(antigen)

    top_sh = df['short_header'].value_counts().index[:5].tolist()
    short_header = df['short_header'].apply(lambda s: s if s in top_sh else "other")
    sh_codes, sh_labels = categorical_colors(short_header)

    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.flat

    # Continuous panels (6)
    continuous = [
        (features['cdr3_length'],    "CDR3 length (residues)",   "viridis"),
        (features['cdr3_mean_phi'],  "CDR3 mean phi (rad)",      "coolwarm"),
        (features['cdr3_mean_psi'],  "CDR3 mean psi (rad)",      "coolwarm"),
        (features['cdr3_var_phi'],   "CDR3 phi variance",        "plasma"),
        (features['cdr3_var_psi'],   "CDR3 psi variance",        "plasma"),
        (seq_lengths.astype(float),  "Total sequence length",    "magma"),
    ]
    for ax, (vals, label, cmap) in zip(axes, continuous):
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=vals, cmap=cmap, s=8, alpha=0.7, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Train / val split
    ax = axes[6]
    split_colors = {'train': '#2196F3', 'val': '#FF5722'}
    for split in ('train', 'val'):
        mask = split_labels == split
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=split_colors[split], s=8, alpha=0.6,
                   label=f"{split} (n={mask.sum()})", rasterized=True)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_title("Train / Val split", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Species
    ax = axes[7]
    for code, label in enumerate(sp_labels):
        mask = sp_codes == code
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap_cat(code)], s=8, alpha=0.8, label=label, rasterized=True)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("Species", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Heavy subclass
    ax = axes[8]
    for code, label in enumerate(sc_labels):
        mask = sc_codes == code
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap_cat(code)], s=8, alpha=0.8, label=label, rasterized=True)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("Heavy subclass (IGHV)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Antigen type
    ax = axes[9]
    for code, label in enumerate(ag_labels):
        mask = ag_codes == code
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap_cat(code)], s=8, alpha=0.8, label=label, rasterized=True)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("Antigen type", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Short header (PDB entry type)
    ax = axes[10]
    for code, label in enumerate(sh_labels):
        mask = sh_codes == code
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap_cat(code)], s=8, alpha=0.8, label=label, rasterized=True)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("PDB entry type (short header)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide unused panel (11 used out of 12)
    axes[11].set_visible(False)

    plt.suptitle(
        f"UMAP of VAE latent space  ({LATENT_DIM}D → 2D)\n"
        f"n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()


# ============================================================================
# PLOT 2 — dimension × CDR3 feature correlation heatmap
# ============================================================================

def plot_dim_correlations(mu_z, features, save_path="dim_correlations.png"):
    keys       = list(features.keys())
    latent_dim = mu_z.shape[1]
    corr       = np.zeros((latent_dim, len(keys)))
    pval       = np.zeros((latent_dim, len(keys)))

    for d in range(latent_dim):
        for j, key in enumerate(keys):
            r, p        = spearmanr(mu_z[:, d], features[key])
            corr[d, j]  = r
            pval[d, j]  = p

    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    plt.colorbar(im, ax=ax, label='Spearman r')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=10)
    ax.set_yticks(range(latent_dim))
    ax.set_yticklabels([f"dim {d}" for d in range(latent_dim)], fontsize=8)

    # Star significant cells
    for d in range(latent_dim):
        for j in range(len(keys)):
            if pval[d, j] < 0.01:
                ax.text(j, d, '*', ha='center', va='center',
                        fontsize=10, color='black')

    ax.set_title(
        "Latent dimension × CDR3 feature correlations\n(Spearman r,  * = p < 0.01)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

    # Print top dims for CDR3 length
    idx  = keys.index("cdr3_length")
    top  = np.argsort(np.abs(corr[:, idx]))[::-1][:8]
    print("\nTop 8 dimensions correlating with CDR3 length (Spearman r):")
    for d in top:
        print(f"  dim {d:2d}:  r = {corr[d, idx]:+.3f},  p = {pval[d, idx]:.4f}")


# ============================================================================
# PLOT 3 — top dimensions vs CDR3 length scatter
# ============================================================================

def plot_top_dims(mu_z, features, n_top=6, save_path="top_dims_cdr3.png"):
    cdr3_length  = features['cdr3_length']
    correlations = [spearmanr(mu_z[:, d], cdr3_length)[0]
                    for d in range(mu_z.shape[1])]
    top_dims     = np.argsort(np.abs(correlations))[::-1][:n_top]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flat

    for ax, d in zip(axes, top_dims):
        r = correlations[d]
        ax.scatter(mu_z[:, d], cdr3_length,
                   c=cdr3_length, cmap='viridis', s=5, alpha=0.6)
        m, b = np.polyfit(mu_z[:, d], cdr3_length, 1)
        xs   = np.linspace(mu_z[:, d].min(), mu_z[:, d].max(), 100)
        ax.plot(xs, m*xs + b, 'r-', linewidth=1.5)
        ax.set_xlabel(f"dim {d}  (mu_z)", fontsize=10)
        ax.set_ylabel("CDR3 length", fontsize=10)
        ax.set_title(f"Dimension {d}   Spearman r = {r:.3f}", fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Top {n_top} latent dimensions vs CDR3 length",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


# ============================================================================
# OUTLIER + SPIKE ANALYSIS
# ============================================================================

def find_outliers(embedding, n_outliers=N_OUTLIERS):
    distances = cdist(embedding, embedding)
    np.fill_diagonal(distances, np.inf)
    min_dist_to_neighbour = distances.min(axis=1)
    outlier_indices = np.argsort(min_dist_to_neighbour)[::-1][:n_outliers]
    return outlier_indices, min_dist_to_neighbour


def find_spike_batch(train_indices, spike_epoch=SPIKE_EPOCH, batch_size=BATCH_SIZE):
    rng = np.random.RandomState(RANDOM_SEED + spike_epoch)
    shuffled = rng.permutation(train_indices)
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


def compute_per_sequence_loss(vae, dataset, device, batch_size=64):
    import torch.distributions as dist_mod

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    all_recon = []

    with torch.no_grad():
        for x, seq_lengths in loader:
            x             = x.to(device)
            seq_lengths   = seq_lengths.to(device)
            _, seq_len, _ = x.shape
            mu_z, lv      = vae.encode(x, seq_lengths)
            std            = (0.5 * lv).exp()
            z              = mu_z + std * torch.randn_like(std)
            mu_x, kappa_x  = vae.decode(z, seq_len)
            mask = (torch.arange(seq_len, device=device)[None, :]
                    < seq_lengths[:, None]).float().unsqueeze(-1)
            recon   = dist_mod.VonMises(mu_x, kappa_x).log_prob(x) * mask
            per_seq = recon.sum(dim=[1, 2]) / seq_lengths.float()
            all_recon.append(per_seq.cpu())

    return torch.cat(all_recon).numpy()


def plot_outlier_spike(embedding, recon_loss, df, outlier_indices,
                       save_path="outlier_spike_analysis.png"):
    cdr3_lengths = np.array([len(df.iloc[i]['cdr3_seq']) for i in range(len(df))])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=recon_loss, cmap='RdYlGn', s=10, alpha=0.8, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Recon loss (lower = better)')
    for rank, idx in enumerate(outlier_indices):
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c='black', s=120, marker='*', zorder=5)
        ax.annotate(f"#{rank+1}\n{df.iloc[idx]['pdb_id']}",
                    (embedding[idx, 0], embedding[idx, 1]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7)
    ax.set_title("UMAP colored by reconstruction loss\n(★ = outlier sequences)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.scatter(cdr3_lengths, recon_loss, s=8, alpha=0.6, c='steelblue')
    for rank, idx in enumerate(outlier_indices):
        ax.scatter(cdr3_lengths[idx], recon_loss[idx],
                   c='red', s=100, marker='*', zorder=5,
                   label=f"#{rank+1} {df.iloc[idx]['pdb_id']}")
    ax.set_xlabel("CDR3 length (residues)", fontsize=11)
    ax.set_ylabel("Reconstruction loss (lower = better)", fontsize=11)
    ax.set_title("CDR3 length vs reconstruction loss", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Outlier sequence analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def report_outliers(df, outlier_indices, isolation_scores, recon_loss,
                    embedding, is_train, train_indices):
    print("\n" + "=" * 60)
    print("STEP 1 — Most isolated sequences in UMAP")
    print("=" * 60)
    for rank, idx in enumerate(outlier_indices):
        row = df.iloc[idx]
        print(f"\n  Rank {rank+1} — Dataset index {idx}")
        print(f"    PDB ID      : {row['pdb_id']}")
        print(f"    Species     : {row['species']}")
        print(f"    IGHV        : {row['heavy_subclass']}")
        print(f"    CDR3 seq    : {row['cdr3_seq']}")
        print(f"    CDR3 length : {len(row['cdr3_seq'])}")
        print(f"    Antigen     : {row['antigen_name']}")
        print(f"    Recon loss  : {recon_loss[idx]:.4f}  (lower = better)")
        print(f"    Isolation   : {isolation_scores[idx]:.4f}  (higher = more isolated)")
        print(f"    In train    : {is_train[idx]}")
        print(f"    UMAP coords : ({embedding[idx,0]:.2f}, {embedding[idx,1]:.2f})")

    print("\n" + "=" * 60)
    print(f"STEP 2 — Spike batch contents at epoch {SPIKE_EPOCH}")
    print("=" * 60)
    batches = find_spike_batch(train_indices, SPIKE_EPOCH)
    print(f"Number of batches at epoch {SPIKE_EPOCH}: {len(batches)}")

    outlier_set = set(outlier_indices.tolist())
    found_any = False
    for batch_num, batch_indices in enumerate(batches):
        overlap = outlier_set.intersection(set(batch_indices.tolist()))
        if overlap:
            found_any = True
            print(f"\n  Batch {batch_num} contains outlier(s): {overlap}")
            print(f"  Indices : {batch_indices.tolist()}")
            print(f"  Losses  : {[f'{recon_loss[i]:.4f}' for i in batch_indices]}")
            print(f"  Mean loss: {recon_loss[batch_indices].mean():.4f}")
            print(f"  Max  loss: {recon_loss[batch_indices].max():.4f}")

    if not found_any:
        print("\n  No outliers found in any batch. Showing hardest batch instead:")
        batch_losses = [recon_loss[b].mean() for b in batches]
        hardest = np.argmin(batch_losses)
        print(f"  Hardest batch: {hardest}")
        for idx in batches[hardest]:
            row = df.iloc[idx]
            print(f"    [{idx}] {row['pdb_id']}  CDR3={row['cdr3_seq']}  "
                  f"len={len(row['cdr3_seq'])}  loss={recon_loss[idx]:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Nanobody VAE UMAP analysis")
    parser.add_argument('--run', type=str, default=None,
                        help="Experiment run name; loads from experiments/{run}/")
    args = parser.parse_args()

    run_name   = args.run or _latest_run()
    run_dir    = os.path.join('experiments', run_name)
    checkpoint = os.path.join(run_dir, 'vae_checkpoint.pt')
    out_dir    = run_dir
    print(f"Run: {run_name}")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading model...")
    vae, train_indices, val_indices = load_model(checkpoint, DEVICE)

    print("\nLoading dataset...")
    df      = pd.read_csv(DATA_FILE)
    summary = (pd.read_csv("nanobody_summary.tsv", sep="\t", usecols=["pdb", "short_header"])
                 .drop_duplicates("pdb")
                 .rename(columns={"pdb": "pdb_id"}))
    df = df.merge(summary, on="pdb_id", how="left")
    df["short_header"] = df["short_header"].fillna("unknown")
    dataset = NanobodyDataset(file=DATA_FILE)

    is_train, is_val, split_labels = build_split_mask(
        len(dataset), train_indices, val_indices
    )


    print("\nEncoding dataset...")
    mu_z, seq_lengths = encode_dataset(vae, dataset, DEVICE)

    # Check sequence length distribution between splits
    print(f"Mean train length: {seq_lengths[is_train].mean():.1f}")
    print(f"Mean val length:   {seq_lengths[is_val].mean():.1f}")
    print(f"Std  train length: {seq_lengths[is_train].std():.1f}")
    print(f"Std  val length:   {seq_lengths[is_val].std():.1f}")

    print("\nExtracting CDR3 features from metadata...")
    features = extract_features(df)

    print("\nRunning UMAP (fit on train, transform all)...")
    embedding = run_umap(mu_z, is_train)

    print("\nPlot 1 — UMAP colorings...")
    plot_umap(embedding, features, df, seq_lengths, split_labels,
              save_path=os.path.join(out_dir, 'umap_colorings.png'))

    print("\nPlot 2 — dimension correlations...")
    plot_dim_correlations(mu_z, features,
                          save_path=os.path.join(out_dir, 'dim_correlations.png'))

    print("\nPlot 3 — top dimensions vs CDR3 length...")
    plot_top_dims(mu_z, features,
                  save_path=os.path.join(out_dir, 'top_dims_cdr3.png'))

    print("\nComputing per-sequence reconstruction loss...")
    recon_loss = compute_per_sequence_loss(vae, dataset, DEVICE)

    print("\nFinding outliers...")
    outlier_indices, isolation_scores = find_outliers(embedding)

    report_outliers(df, outlier_indices, isolation_scores, recon_loss,
                    embedding, is_train, train_indices)

    print("\nPlot 4 — outlier spike analysis...")
    plot_outlier_spike(embedding, recon_loss, df, outlier_indices,
                       save_path=os.path.join(out_dir, 'outlier_spike_analysis.png'))

    print("\nAll done. Look up PDB IDs at https://www.rcsb.org")
    worst_indices = np.argsort(recon_loss)[:10]
    for idx in worst_indices:
       row = df.iloc[idx]
       print(f"{row['pdb_id']}  CDR3={row['cdr3_seq']}  len={len(row['cdr3_seq'])}  loss={recon_loss[idx]:.4f}")
if __name__ == "__main__":
    main()