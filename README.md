# nanobody_
# Nanobody CDR3 VAE

A recurrent Variational Autoencoder for nanobody backbone torsion angles (phi, psi) over the merged FW3+CDR3+FW4 region.

---

## Repository structure

```
├── vae.py                   # Model architecture, training loop, dataset
├── diagnostics.py           # Post-training diagnostics
├── umap_analysis.py         # Latent space visualisation and analysis
├── structural_analysis.py   # Dataset characterisation (Ramachandran, omega, CDR3)
├── prep.py                  # Raw PDB → filtered CSV preprocessing
├── run_pipeline.py          # End-to-end pipeline: train → diagnostics → UMAP
├── experiments/             # Auto-generated run directories (gitignored)
│   └── run_YYYYMMDD_XXXXXX/
│       ├── config.json
│       ├── vae_checkpoint.pt
│       ├── training_curves.png
│       ├── diag_*.png
│       ├── umap_colorings.png
│       └── ...
├── nanobodies_filtered.csv  # Processed dataset (see Data section)
└── nanobody_summary.tsv     # Metadata (species, antigen, subclass, resolution)
```

---

## Installation

```bash
git clone https://github.com/slazynp/nanobody-vae
cd nanobody-vae
pip install torch numpy pandas matplotlib scipy umap-learn biopython
```

---

## Data

The model expects `nanobodies_filtered.csv`. Each row contains:

- `pdb_id` — PDB accession
- `torsions` — full FW3+CDR3+FW4 torsion angles as `[[phi, psi, omega], ...]` in degrees
- `cdr3_torsions`, `fw3_torsions`, `fw4_torsions` — region-specific torsions
- `cdr3_seq`, `fw3_seq`, `fw4_seq` — amino acid sequences
- `species`, `heavy_subclass`, `antigen_type`, `antigen_name` — metadata

To reproduce the dataset from raw PDBs:

```bash
python prep.py
```

Requires Chothia-numbered PDB files in `../project/nanobody_pdbs/chothia/` and `nanobody_summary.tsv`.

---

## Usage

**Full pipeline — train + diagnostics + UMAP:**

```bash
python run_pipeline.py
```

**Quick smoke test:**

```bash
python run_pipeline.py --epochs 10 --skip-umap
```

**Training only:**

```bash
python run_pipeline.py --skip-diag --skip-umap
```

All outputs are saved to `experiments/run_YYYYMMDD_XXXXXX/` alongside a `config.json` recording all hyperparameters.

---

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 300 | Maximum training epochs |
| `--patience` | 20 | Early stopping patience |
| `--batch-size` | 32 | Training batch size |
| `--warmup-epochs` | 60 | Epochs before beta starts annealing |
| `--anneal-end` | 170 | Epoch at which beta reaches max_beta |
| `--max-beta` | 0.01 | Maximum KL weight |
| `--lr` | 5e-4 | Adam learning rate |
| `--weight-decay` | 1e-5 | Adam weight decay |
| `--skip-diag` | — | Skip diagnostics stage |
| `--skip-umap` | — | Skip UMAP analysis stage |

---

## Outputs

**Training:**

| File | Description |
|---|---|
| `config.json` | All hyperparameters for this run |
| `vae_checkpoint.pt` | Best model checkpoint including train/val split indices |
| `training_curves.png` | Loss, KL divergence, beta schedule |

**Diagnostics (`python diagnostics.py` or via pipeline):**

| File | Description |
|---|---|
| `diag_kl_per_dim.png` | Per-dimension KL and encoder mean variance |
| `diag_kappa_dist.png` | Decoder concentration (kappa) distribution |
| `diag_reconstruction.png` | Predicted vs real phi/psi on validation set |
| `diag_latent_traversal.png` | Decoder response to ±3σ traversal of each latent dimension |
| `diag_loss_modes.png` | Train vs val loss in train-mode and eval-mode |
| `diag_report.txt` | Full numerical summary |

**UMAP analysis (`python umap_analysis.py` or via pipeline):**

| File | Description |
|---|---|
| `umap_colorings.png` | Latent space colored by CDR3 features, species, IGHV subclass, antigen type |
| `dim_correlations.png` | Spearman correlation of each latent dimension with CDR3 structural features |
| `top_dims_cdr3.png` | Scatter plots of top dimensions vs CDR3 length |
| `outlier_spike_analysis.png` | Structurally unusual sequences and their latent space position |

**Structural analysis (`python structural_analysis.py` — independent of VAE):**

| File | Description |
|---|---|
| `fig1_global_ramachandran.png` | Global Ramachandran scatter, density, region overlays |
| `fig2_outlier_analysis.png` | Ramachandran outlier identification |
| `fig3_aminoacid_ramachandran.png` | Per-amino-acid Ramachandran plots |
| `fig4_near_zero_analysis.png` | Near-zero phi/psi/omega residue analysis |
| `fig5_omega_analysis.png` | Peptide bond planarity |
| `fig6_position_specific.png` | Position-specific angle distributions per region |
| `fig7_metadata_correlations.png` | Quality and metadata correlations |
| `fig8_cdr3_analysis.png` | CDR3 length distribution and amino acid composition |
| `structural_analysis_report.txt` | Full numerical text report |

---

## Model

```
Prior:      p(z)   = N(0, I)
Posterior:  q(z|x) = N(mu(x), diag(sigma²(x)))
Likelihood: p(x|z) = VonMises(mu_x(z), kappa_x(z))
ELBO:       E_q[log p(x|z) / T] - beta * KL(q||p)
```

Key constants (in `vae.py`):

```python
MAX_LEN    = 80    # max sequence length (FW3+CDR3+FW4)
N_ANGLES   = 2     # phi, psi only (omega excluded — 99.8% trans)
HIDDEN_DIM = 64
LATENT_DIM = 32
```

Region boundaries (Chothia numbering, in `prep.py`):

```python
FW3_START,  FW3_END  = 57,  94
CDR3_START, CDR3_END = 95, 102
FW4_START,  FW4_END  = 103, 113
```
