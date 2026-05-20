import os
import math
import glob
import subprocess
import tempfile
import pandas as pd
from Bio import PDB, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data.IUPACData import protein_letters_3to1_extended

TSV_FILE    = "nanobody_summary.tsv"
PDB_DIR     = "../project/nanobody_pdbs/"
CHOTHIA_DIR = os.path.join(PDB_DIR, "chothia")
MAX_LENGTH  = 140

# Chothia residue boundaries for nanobody heavy chain
FW3_START,  FW3_END  = 57,  94
CDR3_START, CDR3_END = 95,  102
FW4_START,  FW4_END  = 103, 113


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(val, default=None):
    try:
        return float(val) if pd.notnull(val) else default
    except (ValueError, TypeError):
        return default


def _has_break_in_regions(chain, regions):
    """Return True if any chain break touches or spans one of the given Chothia ranges.

    A break between residue A and residue B counts as 'in' a region if:
      - A is inside the region, or
      - B is inside the region, or
      - the gap spans the entire region (A < start and B > end).
    """
    segments = PDB.PPBuilder().build_peptides(chain)
    for i in range(len(segments) - 1):
        last_num  = segments[i][-1].id[1]
        first_num = segments[i + 1][0].id[1]
        for start, end in regions:
            if (start <= last_num  <= end) or \
               (start <= first_num <= end) or \
               (last_num < start and first_num > end):
                return True
    return False


def _extract_region(chain, start, end):
    """Return (residues, sequence) for Chothia residue numbers [start, end].

    Insertion-code residues are included and sorted by (resnum, icode).
    HETATM records (id[0] != ' ') are excluded.
    """
    residues = sorted(
        [r for r in chain.get_residues()
         if r.id[0] == " " and start <= r.id[1] <= end],
        key=lambda r: (r.id[1], r.id[2])
    )
    seq = "".join(
        protein_letters_3to1_extended.get(r.resname.capitalize(), "X")
        for r in residues
    )
    return residues, seq


# ============================================================================
# TORSION ANGLE COMPUTATION
# ============================================================================

def _compute_torsions(chain, residues):
    """Compute (phi, psi, omega) for the interior residues of a given list.

    The first and last residues are skipped because their cross-boundary
    dihedrals involve atoms outside the region.

    Returns a list of (phi, psi, omega) tuples, [] if the region is too
    short to have any interior residues, or None if atom data is missing.
    """
    # Exclude HETATM records so neighbour lookup doesn't land on a water/ligand.
    all_residues = [r for r in chain.get_residues() if r.id[0] == " "]
    res_index    = {r.id: i for i, r in enumerate(all_residues)}

    phis, psis, omegas = [], [], []

    for res in residues[1:-1]:
        idx = res_index.get(res.id)
        if idx is None:
            return None

        phi   = _phi(all_residues, idx)
        psi   = _psi(all_residues, idx)
        omega = _omega(all_residues, idx)

        if phi is None or psi is None or omega is None:
            return None

        phis.append(phi)
        psis.append(psi)
        omegas.append(omega)

    # Empty result means the region had ≤ 2 residues — not a computation failure.
    return list(zip(phis, psis, omegas))


def _get_atom(res, name):
    try:
        return res[name].get_vector()
    except KeyError:
        return None


def _dihedral(a, b, c, d):
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = b1 ** b2
    n2 = b2 ** b3
    if n1.norm() == 0 or n2.norm() == 0:
        return None
    cos_angle = max(-1.0, min(1.0, n1 * n2 / (n1.norm() * n2.norm())))
    angle = math.degrees(math.acos(cos_angle))
    if (n1 ** n2) * b2 < 0:
        angle = -angle
    return angle


def _phi(residues, idx):
    if idx == 0:
        return None
    c_prev = _get_atom(residues[idx - 1], "C")
    n      = _get_atom(residues[idx],     "N")
    ca     = _get_atom(residues[idx],     "CA")
    c      = _get_atom(residues[idx],     "C")
    if any(v is None for v in (c_prev, n, ca, c)):
        return None
    return _dihedral(c_prev, n, ca, c)


def _psi(residues, idx):
    if idx >= len(residues) - 1:
        return None
    n      = _get_atom(residues[idx],     "N")
    ca     = _get_atom(residues[idx],     "CA")
    c      = _get_atom(residues[idx],     "C")
    n_next = _get_atom(residues[idx + 1], "N")
    if any(v is None for v in (n, ca, c, n_next)):
        return None
    return _dihedral(n, ca, c, n_next)


def _omega(residues, idx):
    if idx == 0:
        return None
    ca_prev = _get_atom(residues[idx - 1], "CA")
    c_prev  = _get_atom(residues[idx - 1], "C")
    n       = _get_atom(residues[idx],     "N")
    ca      = _get_atom(residues[idx],     "CA")
    if any(v is None for v in (ca_prev, c_prev, n, ca)):
        return None
    return _dihedral(ca_prev, c_prev, n, ca)


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def extract_nanobody_data(tsv_path, pdb_dir, chothia_dir):
    df = pd.read_csv(tsv_path, sep='\t', encoding='latin-1', on_bad_lines='skip')
    df.columns = df.columns.str.strip()

    parser        = PDB.PDBParser(QUIET=True)
    raw_files     = {os.path.basename(f).lower(): f
                     for f in glob.glob(os.path.join(pdb_dir,     "*.pdb"))}
    chothia_files = {os.path.basename(f).lower(): f
                     for f in glob.glob(os.path.join(chothia_dir, "*.pdb"))}

    records = []

    for _, row in df.iterrows():
        pdb_id = str(row['pdb']).strip().lower()
        hchain = str(row['Hchain']).strip()

        # Skip rows with missing identifiers
        if pdb_id in ("nan", ""):
            continue
        if hchain in ("nan", "na", "none", ""):
            continue

        # ── quality filters ──────────────────────────────────────────────
        try:
            r_free = float(row['r_free']) if pd.notnull(row['r_free']) else 999.0
        except ValueError:
            r_free = 999.0

        try:
            resolution = float(row['resolution']) if pd.notnull(row['resolution']) else 999.0
        except ValueError:
            resolution = 999.0

        if str(row.get('scfv', '')).strip().lower() in ('true', '1', 'yes'):
            continue

        if r_free >= 0.30:
            continue

        filename = f"{pdb_id}.pdb"

        # ── full sequence from raw PDB ────────────────────────────────────
        if filename not in raw_files:
            continue
        try:
            structure = parser.get_structure(pdb_id, raw_files[filename])
            chain     = structure[0][hchain]
            peptides  = PDB.PPBuilder().build_peptides(chain)

            if not peptides:
                continue
            if len(peptides) > 1:
                print(f"  Note: {pdb_id} chain {hchain} has {len(peptides)} segments"
                      f" (chain break outside critical regions — keeping)")

            full_seq = "".join(str(pp.get_sequence()) for pp in peptides)

            if not (0 < len(full_seq) <= MAX_LENGTH):
                continue
        except Exception as e:
            print(f"  Skipping {pdb_id} (sequence error): {e}")
            continue

        # ── Chothia-numbered PDB ──────────────────────────────────────────
        if filename not in chothia_files:
            continue
        try:
            c_chain = parser.get_structure(pdb_id, chothia_files[filename])[0][hchain]
        except Exception as e:
            print(f"  Skipping {pdb_id} (Chothia load error): {e}")
            continue

        # Discard if a chain break falls inside FW3, CDR3, or FW4.
        critical_regions = [(FW3_START, FW3_END), (CDR3_START, CDR3_END), (FW4_START, FW4_END)]
        if _has_break_in_regions(c_chain, critical_regions):
            print(f"  Skipping {pdb_id} chain {hchain}: chain break in FW3/CDR3/FW4")
            continue

        # ── region extraction ─────────────────────────────────────────────
        cdr3_residues, cdr3_seq = _extract_region(c_chain, CDR3_START, CDR3_END)
        fw3_residues,  fw3_seq  = _extract_region(c_chain, FW3_START,  FW3_END)
        fw4_residues,  fw4_seq  = _extract_region(c_chain, FW4_START,  FW4_END)

        if not cdr3_seq:
            continue

        if not fw3_seq:
            print(f"  Warning: {pdb_id} chain {hchain} — FW3 empty (keeping entry)")
        if not fw4_seq:
            print(f"  Warning: {pdb_id} chain {hchain} — FW4 empty (keeping entry)")

        # ── torsion angles ────────────────────────────────────────────────
        cdr3_torsions = _compute_torsions(c_chain, cdr3_residues)
        fw3_torsions  = _compute_torsions(c_chain, fw3_residues)
        fw4_torsions  = _compute_torsions(c_chain, fw4_residues)

        # None means atom data was missing — not just a short/empty region.
        if not cdr3_torsions:   # CDR3 must have computable torsions
            continue
        if fw3_torsions is None:
            print(f"  Skipping {pdb_id} chain {hchain}: FW3 torsion computation failed")
            continue
        if fw4_torsions is None:
            print(f"  Skipping {pdb_id} chain {hchain}: FW4 torsion computation failed")
            continue

        # Merged FW3+CDR3+FW4: cross-junction dihedrals (residues 94, 95, 102, 103)
        # are computed correctly because _compute_torsions indexes the full chain.
        merged_residues = fw3_residues + cdr3_residues + fw4_residues
        torsions        = _compute_torsions(c_chain, merged_residues)

        if torsions is None:
            print(f"  Skipping {pdb_id} chain {hchain}: merged torsion computation failed")
            continue
        if not torsions:
            continue

        records.append({
            "id":             f"{pdb_id}_{hchain}",
            "pdb_id":         pdb_id,
            "species":        str(row.get('heavy_species', '')).strip(),
            "hchain":         hchain,
            "resolution":     resolution,
            "r_free":         r_free,
            "r_factor":       _safe_float(row.get('r_factor')),
            "heavy_subclass": str(row.get('heavy_subclass', '')).strip(),
            "antigen_type":   str(row.get('antigen_type',  '')).strip(),
            "antigen_name":   str(row.get('antigen_name',  '')).strip(),
            "engineered":     str(row.get('engineered',    '')).strip(),
            "full_seq":       full_seq,
            "cdr3_seq":       cdr3_seq,
            "fw3_seq":        fw3_seq,
            "fw4_seq":        fw4_seq,
            "sequences":      fw3_seq + cdr3_seq + fw4_seq,
            "torsions":       torsions,
            "cdr3_torsions":  cdr3_torsions,
            "fw3_torsions":   fw3_torsions,
            "fw4_torsions":   fw4_torsions,
        })

    return records


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_cdr3(records):
    """Keep one entry per unique CDR3 sequence, preferring lower resolution."""
    seen = {}
    for rec in records:
        cdr3 = rec["cdr3_seq"]
        if cdr3 not in seen or rec["resolution"] < seen[cdr3]["resolution"]:
            seen[cdr3] = rec
    return list(seen.values())


def cdhit_deduplicate_cdr3(records, identity=0.9):
    """Cluster CDR3 sequences with CD-HIT; keep one representative per cluster.

    Requires cd-hit (conda install -c bioconda cd-hit).
    identity: sequence identity threshold (0.0–1.0).
    """
    if identity >= 0.90:
        word_size = 5
    elif identity >= 0.88:
        word_size = 4
    elif identity >= 0.85:
        word_size = 3
    else:
        word_size = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        input_fasta   = os.path.join(tmpdir, "cdr3_input.fasta")
        output_prefix = os.path.join(tmpdir, "cdr3_clustered")

        with open(input_fasta, "w") as f:
            for rec in records:
                f.write(f">{rec['id']}\n{rec['cdr3_seq']}\n")

        cmd = [
            "cd-hit",
            "-i", input_fasta,
            "-o", output_prefix,
            "-c", str(identity),
            "-n", str(word_size),
            "-l", "5",   # allow sequences shorter than cd-hit's default minimum (10 aa)
            "-d", "0",   # do not truncate sequence IDs in output
            "-T", "0",   # use all available CPU threads
            "-M", "0",   # no memory cap
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            raise RuntimeError(
                "cd-hit not found — install with: conda install -c bioconda cd-hit"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"cd-hit failed:\n{e.stderr.decode()}")

        representative_ids = {rec.id for rec in SeqIO.parse(output_prefix, "fasta")}

    return [rec for rec in records if rec["id"] in representative_ids]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Extracting sequences and torsion angles...")
    records = extract_nanobody_data(TSV_FILE, PDB_DIR, CHOTHIA_DIR)
    print(f"  {len(records)} entries with valid sequence, CDR3, and torsion angles")

    print("Removing exact CDR3 duplicates...")
    records = deduplicate_cdr3(records)
    print(f"  {len(records)} entries after exact CDR3 deduplication")

    print("Clustering CDR3 sequences with CD-HIT (identity=0.9)...")
    records = cdhit_deduplicate_cdr3(records, identity=0.9)
    print(f"  {len(records)} entries after CD-HIT clustering")

    fasta_records = [
        SeqRecord(
            Seq(r["full_seq"]),
            id=r["id"],
            description=f"cdr3:{r['cdr3_seq']} res:{r['resolution']}"
        )
        for r in records
    ]
    SeqIO.write(fasta_records, "nanobodies_filtered.fasta", "fasta")

    df_out = pd.DataFrame([{
        "id":             r["id"],
        "pdb_id":         r["pdb_id"],
        "species":        r["species"],
        "hchain":         r["hchain"],
        "resolution":     r["resolution"],
        "r_free":         r["r_free"],
        "r_factor":       r["r_factor"],
        "heavy_subclass": r["heavy_subclass"],
        "antigen_type":   r["antigen_type"],
        "antigen_name":   r["antigen_name"],
        "engineered":     r["engineered"],
        "full_seq":       r["full_seq"],
        "cdr3_seq":       r["cdr3_seq"],
        "fw3_seq":        r["fw3_seq"],
        "fw4_seq":        r["fw4_seq"],
        "sequences":      r["sequences"],
        "torsions":       [list(t) for t in r["torsions"]],
        "cdr3_torsions":  [list(t) for t in r["cdr3_torsions"]],
        "fw3_torsions":   [list(t) for t in r["fw3_torsions"]],
        "fw4_torsions":   [list(t) for t in r["fw4_torsions"]],
    } for r in records])

    df_out.to_csv("nanobodies_filtered.csv", index=False)
    print("Done. Output: nanobodies_filtered.fasta, nanobodies_filtered.csv")


if __name__ == "__main__":
    main()
