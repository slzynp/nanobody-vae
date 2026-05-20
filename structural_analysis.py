"""
Comprehensive Structural Analysis of Nanobody Dihedral Angles
=============================================================
Figures generated:
  fig1_global_ramachandran.png    — Global scatter + density + region overlays
  fig2_outlier_analysis.png       — Outlier identification and per-structure counts
  fig3_aminoacid_ramachandran.png — Per-amino-acid Ramachandran plots
  fig4_near_zero_analysis.png     — Near-zero φ/ψ/ω residue analysis
  fig5_omega_analysis.png         — Peptide bond planarity (ω angles)
  fig6_position_specific.png      — Position-specific angle distributions
  fig7_metadata_correlations.png  — Quality & metadata correlations
  fig8_cdr3_analysis.png          — CDR3 length + composition
  structural_analysis_report.txt  — Full numerical text report

Input:  nanobodies_filtered.csv
Output: structural_analysis_output/
"""

import os
import json
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from collections import Counter

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
CSV_PATH           = "nanobodies_filtered.csv"
OUTPUT_DIR         = "structural_analysis_output"
NEAR_ZERO_THRESH   = 30.0   # degrees; angles |x| < this are "near zero"
AMINO_ACIDS        = list("ACDEFGHIKLMNPQRSTVWY")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(name):
    return os.path.join(OUTPUT_DIR, name)

# ─── Ramachandran classification ─────────────────────────────────────────────
# Rectangular approximation of canonical regions (degrees).
# Favored  ≈ well-known secondary structure cores.
# Allowed  ≈ broader energetically accessible regions.
# Outlier  ≈ outside all allowed regions (stereochemically strained).

def _in(phi, psi, phi_lo, phi_hi, psi_lo, psi_hi):
    return phi_lo <= phi <= phi_hi and psi_lo <= psi <= psi_hi

def classify_residue(phi, psi, aa="X"):
    """Return (category, region_name). category: favored|allowed|outlier."""
    if pd.isna(phi) or pd.isna(psi):
        return "missing", "missing"

    # Glycine: no Cβ → much broader allowed region
    if aa == "G":
        if (_in(phi, psi, -90, -30, -80, 5) or
                _in(phi, psi, -170, -50, 50, 180) or
                _in(phi, psi, 20, 100, 20, 90)):
            return "favored", "gly_favored"
        return "allowed", "gly_allowed"

    # Proline: pyrrolidine ring restricts phi to ≈ -65°
    if aa == "P":
        if _in(phi, psi, -85, -40, -60, 10):
            return "favored", "pro_alpha"
        if _in(phi, psi, -85, -40, 90, 180):
            return "favored", "pro_beta"
        if _in(phi, psi, -100, -25, -80, 180):
            return "allowed", "pro_allowed"
        return "outlier", "pro_outlier"

    # General residues — favored cores
    if _in(phi, psi, -90, -30, -80, 5):
        return "favored", "alpha"
    # β-sheet: standard range AND the ±180° wrap (phi≈+175° == phi≈-185°)
    if (_in(phi, psi, -170, -50, 50, 180) or _in(phi, psi, -170, -50, -180, -155) or
            _in(phi, psi, 160, 180, 50, 180) or _in(phi, psi, 160, 180, -180, -155)):
        return "favored", "beta"
    if _in(phi, psi, 20, 100, 20, 90):
        return "favored", "left_alpha"
    if _in(phi, psi, -90, -40, 120, 180):
        return "favored", "ppii"

    # Allowed — generous coverage of the accessible negative-phi half-space
    # and the left-alpha extension; only phi > 0 far from left-alpha is disallowed.
    if phi <= 0:
        # Whole negative-phi half is "allowed" except the narrow strained zone
        # directly around phi≈0, psi≈90 (alpha→beta bridge — truly disfavored).
        if not _in(phi, psi, -10, 0, 50, 130):
            return "allowed", "general"
    # Positive phi: only the left-handed helix extension is allowed
    if _in(phi, psi, 0, 130, -20, 110):
        return "allowed", "pos_phi"

    # Truly strained: positive phi outside all accessible regions, or the
    # narrow strained bridge near (0°, 90°) for negative phi.
    return "outlier", "outlier"


def classify_omega(omega):
    """trans (|ω|>150°) | cis (|ω|<30°) | distorted."""
    if pd.isna(omega):
        return "missing"
    a = abs(omega)
    if a > 150:
        return "trans"
    if a < 30:
        return "cis"
    return "distorted"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _parse_torsions(raw):
    if pd.isna(raw) or str(raw).strip() in ("", "[]"):
        return []
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(str(raw))
        except Exception:
            return []


def load_data(csv_path):
    """
    Returns (df, res_df).
    res_df: one row per residue with phi, psi, omega, aa, region, metadata.
    Torsions are computed for interior residues seq[1:-1] in prep.py.
    """
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        sid        = row["id"]
        species    = row.get("species", "unknown")
        resolution = row.get("resolution", np.nan)
        r_free     = row.get("r_free", np.nan)
        antigen    = row.get("antigen_type", "unknown")
        subclass   = row.get("heavy_subclass", "unknown")

        for region, tcol, scol in [
            ("CDR3", "cdr3_torsions", "cdr3_seq"),
            ("FW3",  "fw3_torsions",  "fw3_seq"),
            ("FW4",  "fw4_torsions",  "fw4_seq"),
        ]:
            torsions  = _parse_torsions(row[tcol])
            seq       = str(row.get(scol, "") or "")
            inner_seq = seq[1:-1] if len(seq) > 2 else ""

            for i, (phi, psi, omega) in enumerate(torsions):
                aa       = inner_seq[i] if i < len(inner_seq) else "X"
                cat, reg = classify_residue(phi, psi, aa)
                rows.append({
                    "structure_id" : sid,
                    "species"      : species,
                    "resolution"   : resolution,
                    "r_free"       : r_free,
                    "antigen_type" : antigen,
                    "heavy_subclass": subclass,
                    "region"       : region,
                    "position"     : i,
                    "aa"           : aa,
                    "phi"          : float(phi),
                    "psi"          : float(psi),
                    "omega"        : float(omega),
                    "rama_cat"     : cat,
                    "rama_region"  : reg,
                    "omega_cat"    : classify_omega(omega),
                })

    return df, pd.DataFrame(rows)


# ─── Plotting helpers ─────────────────────────────────────────────────────────

_RAMA_RECTS = [
    # phi_lo, phi_hi, psi_lo, psi_hi, color, label
    (-90,  -30, -80,   5,  "#3498db", "α-Helix"),
    (-170, -50,  50, 180,  "#e74c3c", "β-Sheet"),
    (-170, -50, -180,-155, "#e74c3c", None),
    (20,   100,  20,  90,  "#2ecc71", "Left-α"),
    (-90,  -40, 120, 180,  "#f39c12", "PPII"),
]

def draw_background(ax):
    for ph0, ph1, ps0, ps1, col, lbl in _RAMA_RECTS:
        ax.add_patch(mpatches.Rectangle(
            (ph0, ps0), ph1-ph0, ps1-ps0,
            linewidth=0, facecolor=col, alpha=0.13,
            label=lbl if lbl else "_nolegend_"
        ))
    ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("φ (°)", fontsize=8)
    ax.set_ylabel("ψ (°)", fontsize=8)


# ─── Figure 1: Global Ramachandran ───────────────────────────────────────────

def fig1_global(res):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Global Ramachandran Analysis — Nanobody Dataset",
                 fontsize=14, fontweight="bold")

    phi_all = res["phi"].dropna().values
    psi_all = res["psi"].dropna().values

    # 1a scatter by category
    ax = axes[0, 0]
    draw_background(ax)
    cmap_cat = {"favored": "#2c3e50", "allowed": "#7f8c8d",
                "outlier": "#e74c3c", "missing": "none"}
    for cat, clr in cmap_cat.items():
        m = res["rama_cat"] == cat
        ax.scatter(res.loc[m, "phi"], res.loc[m, "psi"],
                   s=3, c=clr, alpha=0.5, label=cat, rasterized=True)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("All residues — by Ramachandran category", fontsize=10, fontweight="bold")

    # 1b KDE density
    ax = axes[0, 1]
    draw_background(ax)
    if len(phi_all) > 10:
        try:
            kde = gaussian_kde(np.vstack([phi_all, psi_all]))
            g = np.linspace(-180, 180, 200)
            xx, yy = np.meshgrid(g, g)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=15, cmap="Blues", alpha=0.85)
            ax.contour(xx, yy, zz, levels=8, colors="navy", linewidths=0.3, alpha=0.4)
        except Exception:
            ax.scatter(phi_all, psi_all, s=1, alpha=0.3, c="steelblue")
    ax.set_title("Density (KDE)", fontsize=10, fontweight="bold")

    # 1c colored by region
    ax = axes[0, 2]
    draw_background(ax)
    rcols = {"CDR3": "#e74c3c", "FW3": "#3498db", "FW4": "#2ecc71"}
    for reg, clr in rcols.items():
        m = res["region"] == reg
        ax.scatter(res.loc[m, "phi"], res.loc[m, "psi"],
                   s=3, c=clr, alpha=0.45, label=reg, rasterized=True)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("Colored by structural region", fontsize=10, fontweight="bold")

    # 1d/e/f per-region KDE
    cmaps = {"CDR3": "Reds", "FW3": "Blues", "FW4": "Greens"}
    for col_i, region in enumerate(["CDR3", "FW3", "FW4"]):
        ax = axes[1, col_i]
        sub = res[res["region"] == region]
        ph = sub["phi"].dropna().values
        ps = sub["psi"].dropna().values
        draw_background(ax)
        if len(ph) > 5:
            try:
                kde_r = gaussian_kde(np.vstack([ph, ps]))
                g = np.linspace(-180, 180, 150)
                xx, yy = np.meshgrid(g, g)
                zz = kde_r(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contourf(xx, yy, zz, levels=12, cmap=cmaps[region], alpha=0.85)
            except Exception:
                ax.scatter(ph, ps, s=2, alpha=0.4)
        fav = (sub["rama_cat"] == "favored").mean() * 100
        oul = (sub["rama_cat"] == "outlier").mean() * 100
        ax.set_title(f"{region}  n={len(ph)}\nFavored {fav:.1f}%  Outlier {oul:.1f}%",
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out("fig1_global_ramachandran.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 1] Global Ramachandran")


# ─── Figure 2: Outlier Analysis ───────────────────────────────────────────────

def fig2_outliers(res, df):
    outliers = res[res["rama_cat"] == "outlier"].copy()
    non_out  = res[res["rama_cat"] != "outlier"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Outlier Analysis", fontsize=14, fontweight="bold")

    # 2a: outlier scatter
    ax = axes[0, 0]
    draw_background(ax)
    ax.scatter(non_out["phi"], non_out["psi"], s=2, c="#bdc3c7", alpha=0.25, rasterized=True)
    ax.scatter(outliers["phi"], outliers["psi"], s=22, c="#e74c3c", alpha=0.85, zorder=5,
               edgecolors="darkred", linewidths=0.4)
    ax.set_title(f"Outliers (n={len(outliers)}) on Ramachandran plot",
                 fontsize=10, fontweight="bold")

    # 2b: top 20 structures by outlier count
    ax = axes[0, 1]
    top = outliers.groupby("structure_id").size().sort_values(ascending=False).head(20)
    ax.barh(range(len(top)), top.values, color="#e74c3c", alpha=0.82)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Outlier residue count")
    ax.set_title("Top 20 structures by outlier count", fontsize=10, fontweight="bold")

    # 2c: category rates by region
    ax = axes[0, 2]
    stats = []
    for reg in ["CDR3", "FW3", "FW4"]:
        sub = res[res["region"] == reg]
        stats.append({
            "region":  reg,
            "favored": (sub["rama_cat"] == "favored").mean() * 100,
            "allowed": (sub["rama_cat"] == "allowed").mean() * 100,
            "outlier": (sub["rama_cat"] == "outlier").mean() * 100,
        })
    sdf = pd.DataFrame(stats).set_index("region")
    sdf[["favored", "allowed", "outlier"]].plot(
        kind="bar", ax=ax, color=["#27ae60", "#f39c12", "#e74c3c"], alpha=0.85)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_title("Ramachandran categories by region", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=0)

    # 2d: outlier rate per amino acid
    ax = axes[1, 0]
    aa_out   = outliers["aa"].value_counts()
    total_aa = res["aa"].value_counts()
    rate_aa  = (aa_out / total_aa).dropna()
    rate_aa  = rate_aa[rate_aa.index != "X"].sort_values(ascending=False).head(15)
    ax.bar(range(len(rate_aa)), rate_aa.values * 100, color="#e74c3c", alpha=0.85)
    ax.set_xticks(range(len(rate_aa)))
    ax.set_xticklabels(rate_aa.index, fontsize=9)
    ax.set_ylabel("Outlier rate (%)")
    ax.set_title("Outlier rate by amino acid", fontsize=10, fontweight="bold")

    # 2e: outlier rate vs resolution
    ax = axes[1, 1]
    ss = res.groupby("structure_id").agg(
        n_out=("rama_cat", lambda x: (x == "outlier").sum()),
        n_tot=("rama_cat", "count"),
        res=("resolution", "first"),
    ).reset_index().dropna(subset=["res"])
    ss["rate"] = ss["n_out"] / ss["n_tot"] * 100
    ax.scatter(ss["res"], ss["rate"], s=15, alpha=0.6, c="#8e44ad")
    if len(ss) > 5:
        z = np.polyfit(ss["res"], ss["rate"], 1)
        xl = np.linspace(ss["res"].min(), ss["res"].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), "r--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel("Outlier rate (%)")
    ax.set_title("Outlier rate vs crystal resolution", fontsize=10, fontweight="bold")

    # 2f: outliers by region colored
    ax = axes[1, 2]
    draw_background(ax)
    for reg, clr in {"CDR3": "#e74c3c", "FW3": "#3498db", "FW4": "#2ecc71"}.items():
        m = outliers["region"] == reg
        ax.scatter(outliers.loc[m, "phi"], outliers.loc[m, "psi"],
                   s=28, c=clr, alpha=0.85, label=f"{reg} ({m.sum()})", zorder=5,
                   edgecolors="black", linewidths=0.3)
    ax.set_title("Outliers coloured by region", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out("fig2_outlier_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 2] Outlier analysis")


# ─── Figure 3: Per-amino-acid Ramachandran ────────────────────────────────────

def fig3_per_aa(res):
    present = [aa for aa in AMINO_ACIDS if aa in res["aa"].values]
    ncols = 5
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.6))
    fig.suptitle("Amino Acid–Specific Ramachandran Distributions",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, aa in enumerate(sorted(present)):
        ax = axes[idx]
        sub = res[res["aa"] == aa]
        draw_background(ax)
        n_tot = len(sub)
        n_out = (sub["rama_cat"] == "outlier").sum()
        ax.scatter(sub["phi"], sub["psi"], s=6, alpha=0.6, c="#2c3e50", rasterized=True)
        if n_out > 0:
            outs = sub[sub["rama_cat"] == "outlier"]
            ax.scatter(outs["phi"], outs["psi"], s=30, c="#e74c3c",
                       zorder=6, alpha=0.9, edgecolors="darkred", linewidths=0.4)
        ax.set_title(f"{aa}  n={n_tot}  out={n_out}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("φ", fontsize=7)
        ax.set_ylabel("ψ", fontsize=7)

    for idx in range(len(present), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(out("fig3_aminoacid_ramachandran.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 3] Per-amino-acid Ramachandran")


# ─── Figure 4: Near-zero residue analysis ────────────────────────────────────

def fig4_near_zero(res):
    T = NEAR_ZERO_THRESH
    nphi  = res[res["phi"].abs() < T]
    npsi  = res[res["psi"].abs() < T]
    nboth = res[(res["phi"].abs() < T) & (res["psi"].abs() < T)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Near-Zero Residue Analysis  (threshold ±{T}°)",
                 fontsize=14, fontweight="bold")

    # 4a near-zero phi
    ax = axes[0, 0]
    draw_background(ax)
    ax.scatter(res["phi"], res["psi"], s=2, c="#bdc3c7", alpha=0.25, rasterized=True)
    ax.scatter(nphi["phi"], nphi["psi"], s=22, c="#e67e22", alpha=0.85, zorder=5)
    ax.axvline(-T, color="orange", lw=1.5, ls="--")
    ax.axvline( T, color="orange", lw=1.5, ls="--")
    ax.set_title(f"|φ| < {T}°   (n={len(nphi)})", fontsize=10, fontweight="bold")

    # 4b near-zero psi
    ax = axes[0, 1]
    draw_background(ax)
    ax.scatter(res["phi"], res["psi"], s=2, c="#bdc3c7", alpha=0.25, rasterized=True)
    ax.scatter(npsi["phi"], npsi["psi"], s=22, c="#9b59b6", alpha=0.85, zorder=5)
    ax.axhline(-T, color="purple", lw=1.5, ls="--")
    ax.axhline( T, color="purple", lw=1.5, ls="--")
    ax.set_title(f"|ψ| < {T}°   (n={len(npsi)})", fontsize=10, fontweight="bold")

    # 4c both near zero
    ax = axes[0, 2]
    draw_background(ax)
    ax.scatter(res["phi"], res["psi"], s=2, c="#bdc3c7", alpha=0.2, rasterized=True)
    ax.scatter(nboth["phi"], nboth["psi"], s=45, c="#e74c3c", alpha=0.9, zorder=6,
               edgecolors="black", linewidths=0.5)
    ax.axvline(-T, color="orange", lw=1.2, ls="--", alpha=0.7)
    ax.axvline( T, color="orange", lw=1.2, ls="--", alpha=0.7)
    ax.axhline(-T, color="purple", lw=1.2, ls="--", alpha=0.7)
    ax.axhline( T, color="purple", lw=1.2, ls="--", alpha=0.7)
    ax.set_title(f"|φ| < {T}° AND |ψ| < {T}°   (n={len(nboth)})",
                 fontsize=10, fontweight="bold")

    # 4d near-zero rate per AA
    ax = axes[1, 0]
    near_any = res[(res["phi"].abs() < T) | (res["psi"].abs() < T)]
    aa_near  = near_any["aa"].value_counts()
    total_aa = res["aa"].value_counts()
    rate = (aa_near / total_aa).dropna()
    rate = rate[rate.index != "X"].sort_values(ascending=False).head(15)
    ax.bar(range(len(rate)), rate.values * 100, color="#e67e22", alpha=0.85)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(rate.index, fontsize=9)
    ax.set_ylabel("Near-zero rate (%)")
    ax.set_title("Near-zero rate by amino acid", fontsize=10, fontweight="bold")

    # 4e near-zero counts by region
    ax = axes[1, 1]
    rows_reg = {}
    for reg in ["CDR3", "FW3", "FW4"]:
        sub = res[res["region"] == reg]
        rows_reg[reg] = {
            f"|φ|<{T}°":            (sub["phi"].abs() < T).sum(),
            f"|ψ|<{T}°":            (sub["psi"].abs() < T).sum(),
            f"both<{T}°": ((sub["phi"].abs() < T) & (sub["psi"].abs() < T)).sum(),
        }
    pd.DataFrame(rows_reg).T.plot(
        kind="bar", ax=ax, color=["#e67e22", "#9b59b6", "#e74c3c"], alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title("Near-zero residues by region", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=0)

    # 4f phi and psi distributions with near-zero zone shaded
    ax = axes[1, 2]
    bins = np.linspace(-180, 180, 73)
    ax.hist(res["phi"].dropna(), bins=bins, alpha=0.5, color="#e67e22",
            label="φ", density=True)
    ax.hist(res["psi"].dropna(), bins=bins, alpha=0.5, color="#9b59b6",
            label="ψ", density=True)
    ax.axvspan(-T, T, alpha=0.12, color="gray", label=f"Near-zero zone (±{T}°)")
    ax.axvline(-T, color="gray", lw=1, ls="--")
    ax.axvline( T, color="gray", lw=1, ls="--")
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Density")
    ax.set_title("φ/ψ distributions with near-zero zone", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out("fig4_near_zero_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 4] Near-zero analysis")


# ─── Figure 5: Omega planarity ────────────────────────────────────────────────

def fig5_omega(res):
    omegas = res["omega"].dropna()
    bins   = np.linspace(-180, 180, 73)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ω (Omega) Peptide Bond Planarity Analysis",
                 fontsize=14, fontweight="bold")

    # 5a global distribution
    ax = axes[0, 0]
    ax.hist(omegas, bins=bins, color="#16a085", alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline( 180, color="#c0392b", lw=2, ls="--", label="Trans (±180°)")
    ax.axvline(-180, color="#c0392b", lw=2, ls="--")
    ax.axvline(0,   color="#e67e22", lw=2, ls="--", label="Cis (0°)")
    ax.axvspan(-30, 30, alpha=0.2, color="#e67e22", label="Cis zone")
    ax.set_xlabel("ω (°)")
    ax.set_ylabel("Count")
    ax.set_title("Global ω distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 5b trans distortion (deviation from 180°)
    ax = axes[0, 1]
    trans_o = omegas[omegas.abs() > 90]
    dev     = 180 - trans_o.abs()
    ax.hist(dev, bins=50, color="#2980b9", alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(15, color="#e74c3c", lw=2, ls="--", label="15° threshold")
    pct = (dev > 15).mean() * 100
    ax.text(0.64, 0.93, f"> 15°: {pct:.1f}%", transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel("Deviation from ±180° (°)")
    ax.set_ylabel("Count")
    ax.set_title("Trans peptide bond distortion", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 5c per-region
    ax = axes[1, 0]
    for reg, col in [("CDR3", "#e74c3c"), ("FW3", "#3498db"), ("FW4", "#2ecc71")]:
        ax.hist(res[res["region"] == reg]["omega"].dropna(),
                bins=bins, alpha=0.5, color=col, label=reg, density=True)
    ax.set_xlabel("ω (°)")
    ax.set_ylabel("Density")
    ax.set_title("ω distribution by region", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 5d cis peptides by AA
    ax = axes[1, 1]
    cis = res[res["omega"].abs() < 30]
    if len(cis) > 0:
        cis_rate = (cis["aa"].value_counts() / res["aa"].value_counts()).dropna()
        cis_rate = cis_rate[cis_rate.index != "X"].sort_values(ascending=False).head(12)
        bar_cols  = ["#e74c3c" if aa == "P" else "#2980b9" for aa in cis_rate.index]
        ax.bar(range(len(cis_rate)), cis_rate.values * 100, color=bar_cols, alpha=0.85)
        ax.set_xticks(range(len(cis_rate)))
        ax.set_xticklabels(cis_rate.index, fontsize=9)
        ax.set_ylabel("Cis rate (%)")
        ax.set_title(f"Cis peptides by AA (n={len(cis)}, red=Pro)",
                     fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No cis peptides found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Cis peptides by AA", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out("fig5_omega_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 5] Omega analysis")


# ─── Figure 6: Position-specific ─────────────────────────────────────────────

def fig6_position(res):
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("Position-Specific Dihedral Angle Distributions",
                 fontsize=14, fontweight="bold")

    specs = [
        ("CDR3", "#c0392b", "#e74c3c"),
        ("FW3",  "#1a5276", "#2980b9"),
        ("FW4",  "#145a32", "#27ae60"),
    ]

    for row_i, (region, cphi, cpsi) in enumerate(specs):
        sub  = res[res["region"] == region]
        mpos = int(sub["position"].max()) + 1 if len(sub) else 1

        for col_i, (angle_col, ref_lines, col, ylabel) in enumerate([
            ("phi", [(-57, "#1a6bb5", "α φ"), (-119, "#196f3d", "β φ")], cphi, "φ (°)"),
            ("psi", [(-47, "#1a6bb5", "α ψ"), ( 113, "#196f3d", "β ψ")], cpsi, "ψ (°)"),
        ]):
            ax  = axes[row_i, col_i]
            grp = [sub[sub["position"] == p][angle_col].dropna().values
                   for p in range(mpos)]
            grp = [g for g in grp if len(g) > 0]
            if grp:
                ax.boxplot(grp, positions=range(len(grp)), widths=0.6,
                           patch_artist=True,
                           boxprops=dict(facecolor=col, alpha=0.55),
                           medianprops=dict(color="white", lw=2),
                           whiskerprops=dict(color=col),
                           capprops=dict(color=col),
                           flierprops=dict(marker=".", color="#e74c3c", ms=3, alpha=0.5))
            ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
            for val, lc, lbl in ref_lines:
                ax.axhline(val, color=lc, lw=0.9, ls=":", alpha=0.6, label=lbl)
            ax.set_ylim(-185, 185)
            ax.set_xlabel("Position index", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{region} — {ylabel} per position", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(out("fig6_position_specific.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 6] Position-specific")


# ─── Figure 7: Metadata correlations ─────────────────────────────────────────

def fig7_metadata(res, df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Structural Quality & Metadata Correlations",
                 fontsize=14, fontweight="bold")

    # 7a category by heavy subclass
    ax = axes[0, 0]
    top_sc = res["heavy_subclass"].value_counts().head(6).index
    x = np.arange(len(top_sc))
    for i, (cat, clr) in enumerate([("favored","#27ae60"),("allowed","#f39c12"),("outlier","#e74c3c")]):
        vals = [(res[res["heavy_subclass"]==sc]["rama_cat"]==cat).mean()*100 for sc in top_sc]
        ax.bar(x + i*0.28, vals, 0.28, label=cat, color=clr, alpha=0.85)
    ax.set_xticks(x + 0.28)
    ax.set_xticklabels(top_sc, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Ramachandran by heavy chain subclass", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 7b resolution histogram
    ax = axes[0, 1]
    rc = df["resolution"].dropna()
    ax.hist(rc, bins=30, color="#2980b9", alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(rc.median(), color="#e74c3c", lw=2, ls="--", label=f"Median {rc.median():.2f} Å")
    ax.axvline(rc.mean(),   color="#f39c12", lw=2, ls="--", label=f"Mean {rc.mean():.2f} Å")
    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel("Count")
    ax.set_title("Resolution distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 7c antigen type pie
    ax = axes[0, 2]
    ac = df["antigen_type"].value_counts().head(8)
    ax.pie(ac.values, labels=ac.index, autopct="%1.1f%%",
           colors=plt.cm.Set3(np.linspace(0, 1, len(ac))),
           textprops={"fontsize": 8})
    ax.set_title("Antigen type distribution", fontsize=10, fontweight="bold")

    # 7d species bar
    ax = axes[1, 0]
    sc_cnt = df["species"].value_counts().head(8)
    ax.barh(range(len(sc_cnt)), sc_cnt.values, color="#8e44ad", alpha=0.85)
    ax.set_yticks(range(len(sc_cnt)))
    ax.set_yticklabels([s.title() for s in sc_cnt.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Species distribution", fontsize=10, fontweight="bold")

    # 7e outlier rate vs r_free
    ax = axes[1, 1]
    ss = res.groupby("structure_id").agg(
        n_out=("rama_cat", lambda x: (x=="outlier").sum()),
        n_tot=("rama_cat", "count"),
        rf=("r_free", "first"),
    ).reset_index().dropna(subset=["rf"])
    ss["rate"] = ss["n_out"] / ss["n_tot"] * 100
    ax.scatter(ss["rf"], ss["rate"], s=15, alpha=0.6, c="#16a085")
    if len(ss) > 5:
        z = np.polyfit(ss["rf"], ss["rate"], 1)
        xl = np.linspace(ss["rf"].min(), ss["rf"].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), "r--", lw=1.5, alpha=0.7)
    ax.set_xlabel("R-free")
    ax.set_ylabel("Outlier rate (%)")
    ax.set_title("Outlier rate vs R-free", fontsize=10, fontweight="bold")

    # 7f mean phi/psi by antigen type
    ax = axes[1, 2]
    top_ant = res["antigen_type"].value_counts().head(5).index
    x = np.arange(len(top_ant))
    mphi = [res[res["antigen_type"]==a]["phi"].mean() for a in top_ant]
    mpsi = [res[res["antigen_type"]==a]["psi"].mean() for a in top_ant]
    ax.bar(x - 0.2, mphi, 0.38, label="Mean φ", color="#e67e22", alpha=0.85)
    ax.bar(x + 0.2, mpsi, 0.38, label="Mean ψ", color="#8e44ad", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top_ant, fontsize=8, rotation=30, ha="right")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("Mean angle (°)")
    ax.set_title("Mean φ/ψ by antigen type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out("fig7_metadata_correlations.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 7] Metadata correlations")


# ─── Figure 8: CDR3 length & composition ─────────────────────────────────────

def fig8_cdr3(res, df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("CDR3 Length & Sequence Composition Analysis",
                 fontsize=14, fontweight="bold")

    seqs  = df["cdr3_seq"].dropna().astype(str)
    lens  = seqs.str.len()
    cdr3r = res[res["region"] == "CDR3"]

    # 8a length histogram
    ax = axes[0, 0]
    ax.hist(lens, bins=range(int(lens.min()), int(lens.max())+2),
            color="#c0392b", alpha=0.85, edgecolor="white", lw=0.5, align="left")
    ax.set_xlabel("CDR3 length (AA)")
    ax.set_ylabel("Count")
    ax.set_title(f"CDR3 length distribution\n"
                 f"Mean {lens.mean():.1f}  Med {lens.median():.0f}  "
                 f"Range {lens.min()}–{lens.max()}",
                 fontsize=9, fontweight="bold")

    # 8b AA composition vs proteome average
    ax = axes[0, 1]
    all_aa  = "".join(seqs)
    aa_cnt  = Counter(all_aa)
    total   = len(all_aa)
    aa_ref  = {"A":8.25,"R":5.53,"N":4.06,"D":5.45,"C":1.37,"Q":3.93,
               "E":6.75,"G":7.07,"H":2.27,"I":5.96,"L":9.66,"K":5.84,
               "M":2.42,"F":3.86,"P":4.70,"S":6.56,"T":5.34,"W":1.08,
               "Y":2.92,"V":6.87}
    aas  = sorted(aa_cnt.keys())
    obs  = [aa_cnt[a]/total*100 for a in aas]
    exp  = [aa_ref.get(a, 0)    for a in aas]
    x = np.arange(len(aas))
    ax.bar(x, obs, alpha=0.7, color="#c0392b", label="CDR3 observed")
    ax.plot(x, exp, "ko-", ms=4, lw=1.5, label="Proteome avg")
    ax.set_xticks(x)
    ax.set_xticklabels(aas, fontsize=8)
    ax.set_ylabel("Frequency (%)")
    ax.set_title("CDR3 AA composition vs proteome average", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # 8c CDR3 phi/psi overlay histogram
    ax = axes[0, 2]
    bins = np.linspace(-180, 180, 37)
    ax.hist(cdr3r["phi"].dropna(), bins=bins, color="#c0392b", alpha=0.65,
            label="CDR3 φ", density=True)
    ax.hist(cdr3r["psi"].dropna(), bins=bins, color="#e74c3c", alpha=0.45,
            label="CDR3 ψ", density=True)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Density")
    ax.set_title("CDR3 φ and ψ distributions", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 8d CDR3 length vs CDR3 outlier rate
    ax = axes[1, 0]
    out_rate = cdr3r.groupby("structure_id").agg(
        n_out=("rama_cat", lambda x: (x=="outlier").sum()),
        n_tot=("rama_cat", "count"),
    ).reset_index()
    out_rate["rate"] = out_rate["n_out"] / out_rate["n_tot"] * 100
    merge = df[["id","cdr3_seq"]].copy()
    merge["length"] = merge["cdr3_seq"].str.len()
    m2 = merge.merge(out_rate, left_on="id", right_on="structure_id")
    if len(m2) > 5:
        ax.scatter(m2["length"], m2["rate"], s=15, alpha=0.6, c="#c0392b")
        z = np.polyfit(m2["length"], m2["rate"], 1)
        xl = np.linspace(m2["length"].min(), m2["length"].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), "b--", lw=1.5, alpha=0.7)
    ax.set_xlabel("CDR3 length (AA)")
    ax.set_ylabel("CDR3 outlier rate (%)")
    ax.set_title("CDR3 length vs outlier rate", fontsize=10, fontweight="bold")

    # 8e AA frequency heatmap per CDR3 position
    ax = axes[1, 1]
    max_len = min(int(lens.max()), 25)
    freq    = np.zeros((len(AMINO_ACIDS), max_len))
    for seq in seqs:
        for p, aa in enumerate(seq[:max_len]):
            if aa in AMINO_ACIDS:
                freq[AMINO_ACIDS.index(aa), p] += 1
    col_s = freq.sum(axis=0)
    col_s[col_s == 0] = 1
    freq_n = freq / col_s
    im = ax.imshow(freq_n, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=freq_n.max())
    ax.set_yticks(range(len(AMINO_ACIDS)))
    ax.set_yticklabels(AMINO_ACIDS, fontsize=7)
    ax.set_xlabel("CDR3 position")
    ax.set_title("AA frequency heatmap by CDR3 position", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Frequency")

    # 8f CDR3 length by species
    ax = axes[1, 2]
    top_sp = df["species"].value_counts().head(5).index
    data_s = [df[df["species"]==sp]["cdr3_seq"].str.len().dropna().values for sp in top_sp]
    lbl_s  = [sp.split()[0].title() for sp in top_sp]
    data_s = [(d, l) for d, l in zip(data_s, lbl_s) if len(d) > 0]
    if data_s:
        dvals, dlbls = zip(*data_s)
        ax.boxplot(dvals, labels=dlbls, patch_artist=True,
                   boxprops=dict(facecolor="#c0392b", alpha=0.6),
                   medianprops=dict(color="white", lw=2))
    ax.set_ylabel("CDR3 length (AA)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("CDR3 length by species", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out("fig8_cdr3_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 8] CDR3 analysis")


# ─── Text report ──────────────────────────────────────────────────────────────

def generate_report(res, df):
    T   = NEAR_ZERO_THRESH
    sep = "=" * 70
    L   = []

    def add(s=""):
        L.append(s)

    add(sep)
    add("COMPREHENSIVE STRUCTURAL ANALYSIS REPORT")
    add("Nanobody Backbone Dihedral Angle Analysis")
    add(f"Dataset : {CSV_PATH}")
    add(f"Date    : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(sep)

    # ── 1. Overview
    add("\n1. DATASET OVERVIEW")
    add("-" * 40)
    add(f"  Structures           : {len(df)}")
    add(f"  Total residues       : {len(res)}")
    for reg in ["CDR3","FW3","FW4"]:
        add(f"    {reg:5s}             : {(res['region']==reg).sum()}")
    add(f"  Species              : {df['species'].nunique()}")
    add(f"  Antigen types        : {df['antigen_type'].nunique()}")
    rr = df["resolution"].dropna()
    add(f"  Resolution range     : {rr.min():.2f} – {rr.max():.2f} Å")
    add(f"  Mean resolution      : {rr.mean():.2f} ± {rr.std():.2f} Å")

    # ── 2. Ramachandran
    add("\n2. RAMACHANDRAN STATISTICS")
    add("-" * 40)
    for region in ["CDR3","FW3","FW4","ALL"]:
        sub = res if region == "ALL" else res[res["region"]==region]
        n   = max(len(sub), 1)
        nf  = (sub["rama_cat"]=="favored").sum()
        na  = (sub["rama_cat"]=="allowed").sum()
        no  = (sub["rama_cat"]=="outlier").sum()
        add(f"\n  {region} ({len(sub)} residues):")
        add(f"    Favored : {nf:5d}  ({nf/n*100:5.1f}%)")
        add(f"    Allowed : {na:5d}  ({na/n*100:5.1f}%)")
        add(f"    Outlier : {no:5d}  ({no/n*100:5.1f}%)")

    # ── 3. Outliers
    add("\n3. OUTLIER RESIDUES")
    add("-" * 40)
    outliers = res[res["rama_cat"]=="outlier"].copy()
    add(f"  Total outliers              : {len(outliers)}")
    add(f"  Structures with outliers    : {outliers['structure_id'].nunique()}")
    add(f"  Structures free of outliers : {len(df) - outliers['structure_id'].nunique()}")

    add("\n  Top 20 structures with most outliers:")
    top = outliers.groupby("structure_id").size().sort_values(ascending=False).head(20)
    for sid, cnt in top.items():
        row = df[df["id"]==sid]
        rs  = f"{row['resolution'].values[0]:.2f}Å" if len(row) and not pd.isna(row["resolution"].values[0]) else "N/A"
        add(f"    {sid:20s}  {cnt:3d} outliers  res={rs}")

    add("\n  Outlier rate by amino acid:")
    aa_out   = outliers["aa"].value_counts()
    total_aa = res["aa"].value_counts()
    for aa in sorted(aa_out.index):
        if aa == "X":
            continue
        rate = aa_out[aa] / total_aa.get(aa, 1) * 100
        add(f"    {aa}: {aa_out[aa]:4d} / {total_aa.get(aa,0):5d}  ({rate:5.1f}%)")

    # ── 4. Near-zero
    add(f"\n4. NEAR-ZERO RESIDUES  (|angle| < {T}°)")
    add("-" * 40)
    nphi  = res[res["phi"].abs() < T]
    npsi  = res[res["psi"].abs() < T]
    nboth = res[(res["phi"].abs() < T) & (res["psi"].abs() < T)]
    cis_p = res[res["omega"].abs() < T]
    n_tot = max(len(res), 1)
    add(f"  |φ| < {T}°              : {len(nphi):5d}  ({len(nphi)/n_tot*100:.1f}%)")
    add(f"  |ψ| < {T}°              : {len(npsi):5d}  ({len(npsi)/n_tot*100:.1f}%)")
    add(f"  |φ| < {T}° AND |ψ|<{T}°: {len(nboth):5d}  ({len(nboth)/n_tot*100:.2f}%)")
    add(f"  Cis peptides (|ω|<{T}°) : {len(cis_p):5d}  ({len(cis_p)/n_tot*100:.3f}%)")

    if len(nboth) > 0:
        add(f"\n  Residues with BOTH |φ| and |ψ| < {T}°:")
        for _, r in nboth.iterrows():
            add(f"    {r['structure_id']:20s}  {r['region']:5s}  pos={r['position']:3d}"
                f"  {r['aa']}  φ={r['phi']:8.2f}°  ψ={r['psi']:8.2f}°"
                f"  [{r['rama_cat']}]")

    if len(cis_p) > 0:
        add(f"\n  Cis peptide bonds (|ω| < {T}°):")
        for _, r in cis_p.iterrows():
            add(f"    {r['structure_id']:20s}  {r['region']:5s}  pos={r['position']:3d}"
                f"  {r['aa']}  ω={r['omega']:8.2f}°")

    # ── 5. Omega
    add("\n5. OMEGA (ω) PLANARITY")
    add("-" * 40)
    ov = res["omega"].dropna()
    trans    = (ov.abs() > 150).sum()
    cis_o    = (ov.abs() < 30).sum()
    dist     = ((ov.abs() >= 30) & (ov.abs() <= 150)).sum()
    dev180   = 180 - ov[ov.abs() > 90].abs()
    add(f"  Trans (|ω|>150°)         : {trans:6d}  ({trans/len(ov)*100:.1f}%)")
    add(f"  Cis   (|ω|<30°)          : {cis_o:6d}  ({cis_o/len(ov)*100:.3f}%)")
    add(f"  Distorted (30°–150°)     : {dist:6d}  ({dist/len(ov)*100:.1f}%)")
    add(f"  Mean deviation from 180° : {dev180.mean():.2f}°")
    add(f"  Std  deviation from 180° : {dev180.std():.2f}°")
    add(f"  Trans with >15° distortion: {(dev180>15).sum()}  ({(dev180>15).mean()*100:.1f}%)")

    # ── 6. CDR3 length
    add("\n6. CDR3 LENGTH STATISTICS")
    add("-" * 40)
    cl = df["cdr3_seq"].str.len().dropna()
    add(f"  Count  : {len(cl)}")
    add(f"  Mean   : {cl.mean():.2f}")
    add(f"  Std    : {cl.std():.2f}")
    add(f"  Median : {cl.median():.0f}")
    add(f"  Min    : {int(cl.min())}")
    add(f"  Max    : {int(cl.max())}")
    add(f"  Mode   : {int(cl.mode().values[0])}")

    # ── 7. Per-AA stats
    add("\n7. PER-AMINO-ACID SUMMARY (all regions)")
    add("-" * 40)
    add(f"  {'AA':>2}  {'n':>6}  {'φ mean':>8}  {'φ std':>6}  {'ψ mean':>8}  {'ψ std':>6}  {'outlier%':>9}")
    for aa in sorted(AMINO_ACIDS):
        sub = res[res["aa"]==aa]
        if len(sub) == 0:
            continue
        add(f"  {aa:>2}  {len(sub):>6d}"
            f"  {sub['phi'].mean():>8.2f}  {sub['phi'].std():>6.1f}"
            f"  {sub['psi'].mean():>8.2f}  {sub['psi'].std():>6.1f}"
            f"  {(sub['rama_cat']=='outlier').mean()*100:>8.1f}%")

    add("\n" + sep)
    add("END OF REPORT")
    add(sep)

    path = out("structural_analysis_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"[Report] Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Nanobody Comprehensive Structural Analysis")
    print("=" * 60)
    print(f"Loading {CSV_PATH} ...")
    df, res = load_data(CSV_PATH)
    print(f"  {len(df)} structures  |  {len(res)} residues (CDR3+FW3+FW4)\n")

    fig1_global(res)
    fig2_outliers(res, df)
    fig3_per_aa(res)
    fig4_near_zero(res)
    fig5_omega(res)
    fig6_position(res)
    fig7_metadata(res, df)
    fig8_cdr3(res, df)
    generate_report(res, df)

    print("\n" + "=" * 60)
    print(f"All outputs → {OUTPUT_DIR}/")
    print("  fig1_global_ramachandran.png    global scatter + KDE + region split")
    print("  fig2_outlier_analysis.png       outlier locations + per-structure counts")
    print("  fig3_aminoacid_ramachandran.png per-AA Ramachandran (20 panels)")
    print("  fig4_near_zero_analysis.png     |φ|,|ψ| < 30° residues")
    print("  fig5_omega_analysis.png         peptide bond planarity")
    print("  fig6_position_specific.png      position-wise boxplots CDR3/FW3/FW4")
    print("  fig7_metadata_correlations.png  resolution, species, antigen, r_free")
    print("  fig8_cdr3_analysis.png          CDR3 length + AA composition heatmap")
    print("  structural_analysis_report.txt  full numerical report")
    print("=" * 60)


if __name__ == "__main__":
    main()
