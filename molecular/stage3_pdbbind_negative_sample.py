
import os
import glob
import math
import argparse
import shutil
import gzip
import traceback
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

TMP_DIR = "/tmp/rosetta_inputs"


# ============================================================
# 0) Path utils
# ============================================================

def sanitize_path(p: str) -> str:
    p = str(p).strip()
    for _ in range(3):
        p = p.strip().strip('"').strip("'").strip()
    return p


def ensure_tmp_dir() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)


def safe_copy_or_decompress_to_tmp(src_path: str) -> str:
    ensure_tmp_dir()
    src_path = sanitize_path(src_path)
    if not os.path.isfile(src_path):
        return ""

    base = os.path.basename(src_path).replace(" ", "_").replace('"', "").replace("'", "")
    if src_path.endswith(".gz"):
        if base.endswith(".gz"):
            base = base[:-3]
        tmp_path = os.path.join(TMP_DIR, base)
        with gzip.open(src_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return tmp_path
    else:
        tmp_path = os.path.join(TMP_DIR, base)
        shutil.copy2(src_path, tmp_path)
        return tmp_path


# ============================================================
# 1) Table loader (SKEMPI / PDBbind)
# ============================================================

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        c = low.get(cand.lower())
        if c is not None:
            return c
    return None


def load_input_table(path: str) -> Tuple[pd.DataFrame, str]:
    """
    Return (df, mode) where mode in {'skempi','pdbbind'}.
    Auto-detect header row for Excel like PDBbind-CN (first row is a banner).
    """
    path = sanitize_path(path)
    assert os.path.isfile(path), f"input table not found: {path}"

    ext = os.path.splitext(path)[1].lower()

    def _post(df: pd.DataFrame) -> pd.DataFrame:
        df = _norm_cols(df)
        # drop fully unnamed columns if any
        keep = []
        for c in df.columns:
            cs = str(c).strip()
            if cs.lower().startswith("unnamed"):
                continue
            keep.append(c)
        if keep:
            df = df[keep]
        return df

    if ext in [".xlsx", ".xls"]:
        # 1) scan first rows with header=None to locate the real header row
        preview = pd.read_excel(path, header=None, nrows=30, engine="openpyxl")
        header_row = None
        targets = {"#pdb", "pdb code", "pdb_code", "pdb", "pdbcode", "pdb_id"}

        for i in range(preview.shape[0]):
            row_vals = preview.iloc[i].astype(str).str.strip().str.lower().tolist()
            if any(v in targets for v in row_vals):
                header_row = i
                break

        # 2) fallback: common case for your file is header at row 1 (second line)
        if header_row is None:
            header_row = 1

        df = pd.read_excel(path, header=header_row, engine="openpyxl")
        df = _post(df)

    else:
        # auto sep for csv/tsv; SKEMPI sometimes uses ';'
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, sep=";")
        df = _post(df)

    # detect mode
    if _find_col(df, ["#Pdb"]) is not None:
        return df, "skempi"
    if _find_col(df, ["PDB code", "PDB", "PDBCODE", "pdb_id", "PDB_ID"]) is not None:
        return df, "pdbbind"

    raise ValueError(
        f"Cannot detect table mode. Need '#Pdb' (SKEMPI) or 'PDB code' (PDBbind). cols={df.columns.tolist()}"
    )



def parse_skempi_pdb_field(pdb_str: str) -> Tuple[str, str, str]:
    parts = str(pdb_str).split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected #Pdb format: {pdb_str}")
    return parts[0], parts[1], parts[2]


# ============================================================
# 2) PDB locate (support *_complex.pdb)
# ============================================================

def find_pdb_file_robust(pdb_root: str, pdb_id: str) -> Optional[str]:
    pdb_root = sanitize_path(pdb_root)
    pidU = pdb_id.upper()
    pidL = pdb_id.lower()

    patterns = [
        # PDBbind flat format (your case)
        f"{pidL}_complex.pdb", f"{pidU}_complex.pdb",
        f"{pidL}*_complex.pdb", f"{pidU}*_complex.pdb",

        # common pdb names
        f"{pidU}.pdb",  f"{pidL}.pdb",
        f"{pidU}.pdb1", f"{pidL}.pdb1",
        f"{pidU}.ent",  f"{pidL}.ent",
        f"pdb{pidL}.ent", f"pdb{pidU}.ent",
        f"{pidU}*.pdb", f"{pidL}*.pdb",
        f"{pidU}*.ent", f"{pidL}*.ent",
    ]

    for pat in patterns:
        hits = glob.glob(os.path.join(pdb_root, pat))
        if hits:
            hits.sort(key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]

    for pat in patterns:
        hits = glob.glob(os.path.join(pdb_root, pat + ".gz"))
        if hits:
            hits.sort(key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]

    return None


# ============================================================
# 3) PyRosetta init & scoring
# ============================================================

def init_pyrosetta(flags: str):
    import pyrosetta
    # PyRosetta can only init once; make this idempotent
    try:
        pyrosetta.init(options=flags)
    except Exception as e:
        msg = str(e).lower()
        if "already" not in msg and "initialized" not in msg:
            raise
    from pyrosetta import rosetta
    scorefxn = pyrosetta.get_fa_scorefxn()
    return pyrosetta, rosetta, scorefxn


def pose_from_pdb_fixed(pyrosetta, pdb_path: str, debug: bool = False):
    pdb_path = sanitize_path(pdb_path)
    if debug:
        print("[DEBUG] pose_from_pdb input repr:", repr(pdb_path))
    tmp_path = safe_copy_or_decompress_to_tmp(pdb_path)
    if not tmp_path:
        raise FileNotFoundError(f"File not found or not readable: {pdb_path}")
    if debug:
        print("[DEBUG] pose_from_pdb tmp repr:", repr(tmp_path))
    return pyrosetta.pose_from_pdb(tmp_path)


def compute_binding_energy_simple(rosetta, scorefxn, pose, rec_chains: str, lig_chains: str) -> float:
    from pyrosetta import Pose
    from pyrosetta.rosetta.core.pose import pdbslice

    pdb_info = pose.pdb_info()
    rec_set = set(rec_chains)
    lig_set = set(lig_chains)

    rec_res = []
    lig_res = []
    for i in range(1, pose.total_residue() + 1):
        ch = pdb_info.chain(i)
        if ch in rec_set:
            rec_res.append(i)
        elif ch in lig_set:
            lig_res.append(i)

    if not rec_res or not lig_res:
        return float("nan")

    v_rec = rosetta.utility.vector1_unsigned_long()
    for r in rec_res:
        v_rec.append(r)

    v_lig = rosetta.utility.vector1_unsigned_long()
    for r in lig_res:
        v_lig.append(r)

    pose_rec = Pose()
    pdbslice(pose_rec, pose, v_rec)

    pose_lig = Pose()
    pdbslice(pose_lig, pose, v_lig)

    e_complex = scorefxn(pose)
    e_rec = scorefxn(pose_rec)
    e_lig = scorefxn(pose_lig)
    return float(e_complex - (e_rec + e_lig))


# ============================================================
# 4) Infer rec/lig chains for PDBbind (no chain labels in table)
# ============================================================

def infer_rec_lig_from_pose(pose, chain_mode: str = "auto", auto_top2_frac: float = 0.60) -> Optional[Tuple[str, str]]:
    """
    Heuristic chain split:
      - 2 chains: largest -> rec, other -> lig
      - >=3 chains:
          if (top1+top2)/total >= auto_top2_frac: rec=top2, lig=rest
          else: rec=top1, lig=rest
    """
    pdb_info = pose.pdb_info()
    chain_to_res = {}
    for i in range(1, pose.total_residue() + 1):
        ch = pdb_info.chain(i)
        if ch == "" or ch.isspace():
            continue
        chain_to_res[ch] = chain_to_res.get(ch, 0) + 1

    if len(chain_to_res) < 2:
        return None

    chains_sorted = sorted(chain_to_res.keys(), key=lambda c: chain_to_res[c], reverse=True)
    n = len(chains_sorted)

    if chain_mode == "two_chains":
        if n != 2:
            return None
        return chains_sorted[0], chains_sorted[1]

    if chain_mode == "largest1_rest":
        return chains_sorted[0], "".join(chains_sorted[1:])

    if chain_mode == "largest2_rest":
        if n == 2:
            return chains_sorted[0], chains_sorted[1]
        return "".join(chains_sorted[:2]), "".join(chains_sorted[2:])

    # auto
    if n == 2:
        return chains_sorted[0], chains_sorted[1]

    total = sum(chain_to_res[c] for c in chains_sorted)
    top2 = chain_to_res[chains_sorted[0]] + chain_to_res[chains_sorted[1]]
    frac12 = top2 / max(1, total)

    if frac12 >= auto_top2_frac:
        rec = "".join(chains_sorted[:2])
        lig = "".join(chains_sorted[2:])
    else:
        rec = chains_sorted[0]
        lig = "".join(chains_sorted[1:])

    if lig == "":
        return None
    return rec, lig


# ============================================================
# 5) Transform ligand
# ============================================================

def transform_ligand_in_pose(pose, rosetta, lig_chains: str, R: np.ndarray, t: np.ndarray):
    pdb_info = pose.pdb_info()
    lig_set = set(lig_chains)

    atom_ids = []
    coords = []
    for res_i in range(1, pose.total_residue() + 1):
        ch = pdb_info.chain(res_i)
        if ch not in lig_set:
            continue
        res = pose.residue(res_i)
        for atom_idx in range(1, res.natoms() + 1):
            atom_id = rosetta.core.id.AtomID(atom_idx, res_i)
            xyz = pose.xyz(atom_id)
            coords.append([xyz.x, xyz.y, xyz.z])
            atom_ids.append(atom_id)

    if not atom_ids:
        return

    coords = np.asarray(coords, dtype=np.float64)
    com = coords.mean(axis=0)
    rel = coords - com[None, :]
    new_coords = (R @ rel.T).T + com[None, :] + t.reshape(1, 3)

    for atom_id, c in zip(atom_ids, new_coords):
        pose.set_xyz(atom_id, rosetta.numeric.xyzVector_double_t(float(c[0]), float(c[1]), float(c[2])))


# ============================================================
# 6) Random R,t
# ============================================================

def random_rotation_matrix_np() -> np.ndarray:
    u1, u2, u3 = np.random.rand(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)

    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def sample_shift_np(neg_shift_min: float, neg_shift_max: float) -> np.ndarray:
    d = np.random.randn(3)
    n = np.linalg.norm(d)
    if n < 1e-8:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        d = d / n
    mag = np.random.uniform(neg_shift_min, neg_shift_max)
    return d * mag


# ============================================================
# 7) Main precompute (with failures logging)
# ============================================================

def precompute_rosetta_poses(
    skempi_csv: str,
    pdb_root: str,
    out_csv: str,
    n_neg_per_pos: int = 5,
    neg_shift_min: float = 20.0,
    neg_shift_max: float = 40.0,
    seed: int = 2025,
    debug_path: bool = False,
    retry_relaxed: bool = False,
    chain_mode: str = "auto",
    auto_top2_frac: float = 0.60,
    neg_label_mode: str = "zero",  # 'zero' recommended; 'raw' not recommended
):
    skempi_csv = sanitize_path(skempi_csv)
    pdb_root = sanitize_path(pdb_root)
    out_csv = sanitize_path(out_csv)

    assert os.path.isfile(skempi_csv), f"input table not found: {skempi_csv}"
    assert os.path.isdir(pdb_root), f"pdb_root not found: {pdb_root}"

    np.random.seed(seed)

    flags_main = "-mute all -ignore_unrecognized_res -ignore_waters -load_PDB_components false -detect_disulf false"
    flags_relaxed = flags_main

    pyrosetta, rosetta, scorefxn = init_pyrosetta(flags_main)

    print(f"[Precompute] input_table = {skempi_csv}")
    print(f"[Precompute] PDB root    = {pdb_root}")
    print(f"[Precompute] out_csv     = {out_csv}")
    print(f"[Precompute] n_neg_per_pos={n_neg_per_pos}, neg_shift=[{neg_shift_min},{neg_shift_max}] seed={seed}")
    print(f"[Precompute] tmp_dir={TMP_DIR}")
    print(f"[Precompute] retry_relaxed={retry_relaxed}")
    print(f"[Precompute] chain_mode={chain_mode} auto_top2_frac={auto_top2_frac}")
    print(f"[Precompute] neg_label_mode={neg_label_mode}")

    df, mode = load_input_table(skempi_csv)
    print(f"[Precompute] detected mode = {mode}, rows={len(df)}")

    # PDBbind: dedup by PDB code
    if mode == "pdbbind":
        col_pdb = _find_col(df, ["PDB code", "PDB", "PDBCODE", "pdb_id", "PDB_ID"])
        assert col_pdb is not None
        df[col_pdb] = df[col_pdb].astype(str).str.strip().str.slice(0, 4)
        df = df[df[col_pdb].str.len() == 4].drop_duplicates(subset=[col_pdb]).reset_index(drop=True)

    rows_out: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []

    def log_failure(step: str, row_idx: int, pdb_str: str, pdb_id: str,
                    rec_chains: str, lig_chains: str, pdb_file: str, err: Exception):
        tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        fail_rows.append(dict(
            row_idx=row_idx,
            pdb_str=pdb_str,
            pdb_id=pdb_id,
            rec_chains=rec_chains,
            lig_chains=lig_chains,
            step=step,
            pdb_file=pdb_file,
            error_type=type(err).__name__,
            error_msg=str(err),
            traceback=tb[:4000],
        ))
        print(f"[FAIL] row={row_idx} pdb_id={pdb_id} step={step} err={type(err).__name__}: {err}")

    total = len(df)
    skipped_missing_pdb = 0
    skipped_parse_fail = 0
    rosetta_fail = 0
    chain_infer_fail = 0

    for row_idx, row in df.iterrows():
        if mode == "skempi":
            c = _find_col(df, ["#Pdb"])
            pdb_str = str(row[c])
            try:
                pdb_id, rec_chains, lig_chains = parse_skempi_pdb_field(pdb_str)
                pdb_id = str(pdb_id).strip()[:4]
            except Exception as e:
                skipped_parse_fail += 1
                log_failure("parse_pdb_field", row_idx, pdb_str, "NA", "", "", "", e)
                continue
        else:
            col_pdb = _find_col(df, ["PDB code", "PDB", "PDBCODE", "pdb_id", "PDB_ID"])
            pdb_id = str(row[col_pdb]).strip()[:4]
            pdb_str = pdb_id
            rec_chains, lig_chains = "", ""

        pdb_file = find_pdb_file_robust(pdb_root, pdb_id)
        if pdb_file is None:
            skipped_missing_pdb += 1
            continue
        pdb_file = sanitize_path(pdb_file)

        # --- load pose
        try:
            pose = pose_from_pdb_fixed(pyrosetta, pdb_file, debug=debug_path)
        except Exception as e:
            rosetta_fail += 1
            log_failure("pose_from_pdb", row_idx, pdb_str, pdb_id, rec_chains, lig_chains, pdb_file, e)
            if retry_relaxed:
                try:
                    pyrosetta2, rosetta2, scorefxn2 = init_pyrosetta(flags_relaxed)
                    pose = pose_from_pdb_fixed(pyrosetta2, pdb_file, debug=debug_path)
                    rosetta, scorefxn = rosetta2, scorefxn2
                except Exception as e2:
                    log_failure("pose_from_pdb_relaxed", row_idx, pdb_str, pdb_id, rec_chains, lig_chains, pdb_file, e2)
                    continue
            else:
                continue

        # --- infer chains for PDBbind
        if mode == "pdbbind":
            infer = infer_rec_lig_from_pose(pose, chain_mode=chain_mode, auto_top2_frac=auto_top2_frac)
            if infer is None:
                chain_infer_fail += 1
                log_failure("infer_rec_lig", row_idx, pdb_str, pdb_id, "", "", pdb_file,
                            ValueError("Cannot infer rec/lig chains from pose"))
                continue
            rec_chains, lig_chains = infer
            pdb_str = f"{pdb_id}_{rec_chains}_{lig_chains}"

        # --- pos energy
        try:
            dG_pos = compute_binding_energy_simple(rosetta, scorefxn, pose, rec_chains, lig_chains)
        except Exception as e:
            rosetta_fail += 1
            log_failure("energy_pos", row_idx, pdb_str, pdb_id, rec_chains, lig_chains, pdb_file, e)
            continue

        rows_out.append(dict(
            row_idx=row_idx,
            pdb_str=pdb_str,
            pdb_id=pdb_id,
            rec_chains=rec_chains,
            lig_chains=lig_chains,
            pose_type="pos",
            neg_id=0,
            R_flat=",".join(str(x) for x in np.eye(3, dtype=np.float64).ravel()),
            shift="0.0,0.0,0.0",
            dG_bind=float(dG_pos),
            dG_rosetta=float(dG_pos),
        ))

        # --- neg energies
        for neg_id in range(n_neg_per_pos):
            pose_neg = pose.clone()
            R = random_rotation_matrix_np()
            t = sample_shift_np(neg_shift_min, neg_shift_max)
            try:
                transform_ligand_in_pose(pose_neg, rosetta, lig_chains, R, t)
                dG_neg = compute_binding_energy_simple(rosetta, scorefxn, pose_neg, rec_chains, lig_chains)
            except Exception as e:
                rosetta_fail += 1
                log_failure(f"energy_neg_{neg_id}", row_idx, pdb_str, pdb_id, rec_chains, lig_chains, pdb_file, e)
                continue

            dG_bind_neg = float(dG_neg) if neg_label_mode == "raw" else 0.0

            rows_out.append(dict(
                row_idx=row_idx,
                pdb_str=pdb_str,
                pdb_id=pdb_id,
                rec_chains=rec_chains,
                lig_chains=lig_chains,
                pose_type="neg",
                neg_id=neg_id,
                R_flat=",".join(str(x) for x in R.ravel()),
                shift=",".join(str(x) for x in t),
                dG_bind=float(dG_bind_neg),
                dG_rosetta=float(dG_neg),
            ))

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == total:
            print(f"[Precompute] processed {row_idx+1}/{total} | saved={len(rows_out)} "
                  f"| missing_pdb={skipped_missing_pdb} parse_fail={skipped_parse_fail} "
                  f"| chain_infer_fail={chain_infer_fail} rosetta_fail={rosetta_fail}")

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    fail_path = out_csv + ".failures.csv"
    pd.DataFrame(fail_rows).to_csv(fail_path, index=False)

    print(f"[Precompute] DONE. saved_rows={len(out_df)} -> {out_csv}")
    print(f"[Precompute] FAILURES n={len(fail_rows)} -> {fail_path}")
    print(f"[Precompute] missing_pdb={skipped_missing_pdb}, parse_fail={skipped_parse_fail}, "
          f"chain_infer_fail={chain_infer_fail}, rosetta_fail={rosetta_fail}")


# ============================================================
# 8) CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser("Precompute Rosetta binding energies for SKEMPI/PDBbind (fixed for PDBbind).")
    ap.add_argument("--skempi_csv", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/PDBbind-CN_v2020_PP_20231108.xlsx",
                    help="SKEMPI CSV (with #Pdb) OR PDBbind Excel (with PDB code).")
    ap.add_argument("--pdb_root", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/PDBbind",
                    help="Directory containing PDB files (e.g. 1a3b_complex.pdb)")
    ap.add_argument("--out_csv", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/stage3_negative_sample_pdbbind.csv",
                    help="Output CSV path")

    ap.add_argument("--n_neg_per_pos", type=int, default=5)
    ap.add_argument("--neg_shift_min", type=float, default=20.0)
    ap.add_argument("--neg_shift_max", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--debug_path", action="store_true")
    ap.add_argument("--retry_relaxed", action="store_true")

    ap.add_argument("--chain_mode", type=str, default="auto",
                    choices=["auto", "two_chains", "largest1_rest", "largest2_rest"])
    ap.add_argument("--auto_top2_frac", type=float, default=0.60)

    ap.add_argument("--neg_label_mode", type=str, default="zero",
                    choices=["zero", "raw"],
                    help="zero: neg dG_bind=0 (recommended); raw: neg dG_bind=dG_rosetta (NOT recommended)")

    args = ap.parse_args()

    precompute_rosetta_poses(
        skempi_csv=args.skempi_csv,
        pdb_root=args.pdb_root,
        out_csv=args.out_csv,
        n_neg_per_pos=args.n_neg_per_pos,
        neg_shift_min=args.neg_shift_min,
        neg_shift_max=args.neg_shift_max,
        seed=args.seed,
        debug_path=args.debug_path,
        retry_relaxed=args.retry_relaxed,
        chain_mode=args.chain_mode,
        auto_top2_frac=args.auto_top2_frac,
        neg_label_mode=args.neg_label_mode,
    )


if __name__ == "__main__":
    main()
