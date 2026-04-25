# -*- coding: utf-8 -*-
"""
Precompute Rosetta binding energies for SKEMPI positive + negative poses (FIXED v2).

新增能力：
  - 记录每次 rosetta_fail 的详细信息到 out_csv.failures.csv
  - 控制台打印每个失败样本的关键摘要
  - 可选：对失败样本用更宽松 flags 再尝试读入（--retry_relaxed）

仍然保留：
  - sanitize 路径（去引号/空白）
  - robust 找 PDB（pdb/pdb1/ent + 大小写 + glob + .gz）
  - 所有 PDB copy/decompress 到 /tmp/rosetta_inputs 再喂给 PyRosetta（规避空格/奇怪字符）
"""

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
# 1) SKEMPI parse & PDB locate
# ============================================================

def parse_skempi_pdb_field(pdb_str: str) -> Tuple[str, str, str]:
    parts = str(pdb_str).split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected #Pdb format: {pdb_str}")
    return parts[0], parts[1], parts[2]


def find_pdb_file_robust(pdb_root: str, pdb_id: str) -> Optional[str]:
    pdb_root = sanitize_path(pdb_root)
    pidU = pdb_id.upper()
    pidL = pdb_id.lower()

    patterns = [
        f"{pidU}.pdb",  f"{pidL}.pdb",
        f"{pidU}.pdb1", f"{pidL}.pdb1",
        f"{pidU}.ent",  f"{pidL}.ent",
        f"pdb{pidL}.ent", f"pdb{pidU}.ent",
        f"{pidU}*.pdb", f"{pidL}*.pdb",
        f"{pidU}*.ent", f"{pidL}*.ent",
    ]

    # uncompressed
    for pat in patterns:
        hits = glob.glob(os.path.join(pdb_root, pat))
        if hits:
            hits.sort(key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]

    # gz
    for pat in patterns:
        hits = glob.glob(os.path.join(pdb_root, pat + ".gz"))
        if hits:
            hits.sort(key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]

    return None


# ============================================================
# 2) PyRosetta init & scoring
# ============================================================

def init_pyrosetta(flags: str):
    import pyrosetta
    pyrosetta.init(options=flags)
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
# 3) Transform ligand
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
# 4) Random R,t
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
# 5) Main precompute (with failures logging)
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
):
    skempi_csv = sanitize_path(skempi_csv)
    pdb_root = sanitize_path(pdb_root)
    out_csv = sanitize_path(out_csv)

    assert os.path.isfile(skempi_csv), f"SKEMPI csv not found: {skempi_csv}"
    assert os.path.isdir(pdb_root), f"pdb_root not found: {pdb_root}"

    np.random.seed(seed)

    # 两套 flags：标准 + 更宽松（可选）
    flags_main = "-mute all -ignore_unrecognized_res -detect_disulf false"
    flags_relaxed = "-mute all -ignore_unrecognized_res -ignore_waters -load_PDB_components false -detect_disulf false"

    pyrosetta, rosetta, scorefxn = init_pyrosetta(flags_main)

    print(f"[Precompute] SKEMPI csv = {skempi_csv}")
    print(f"[Precompute] PDB root   = {pdb_root}")
    print(f"[Precompute] out_csv    = {out_csv}")
    print(f"[Precompute] n_neg_per_pos={n_neg_per_pos}, neg_shift=[{neg_shift_min},{neg_shift_max}] seed={seed}")
    print(f"[Precompute] tmp_dir={TMP_DIR}")
    print(f"[Precompute] retry_relaxed={retry_relaxed}")

    df = pd.read_csv(skempi_csv, sep=";")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce").fillna(298.0)
    if "Affinity_mut_parsed" in df.columns and "Affinity_wt_parsed" in df.columns:
        df = df[(df["Affinity_mut_parsed"] > 0) & (df["Affinity_wt_parsed"] > 0)].reset_index(drop=True)

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

    for row_idx, row in df.iterrows():
        pdb_str = row["#Pdb"]

        try:
            pdb_id, rec_chains, lig_chains = parse_skempi_pdb_field(pdb_str)
        except Exception as e:
            skipped_parse_fail += 1
            log_failure("parse_pdb_field", row_idx, pdb_str, "NA", "", "", "", e)
            continue

        pdb_file = find_pdb_file_robust(pdb_root, pdb_id)
        if pdb_file is None:
            skipped_missing_pdb += 1
            continue
        pdb_file = sanitize_path(pdb_file)

        if debug_path:
            print(f"[DEBUG] pdb_file repr={repr(pdb_file)} exists={os.path.isfile(pdb_file)}")

        # --- load pose
        try:
            pose = pose_from_pdb_fixed(pyrosetta, pdb_file, debug=debug_path)
        except Exception as e:
            rosetta_fail += 1
            log_failure("pose_from_pdb", row_idx, pdb_str, pdb_id, rec_chains, lig_chains, pdb_file, e)

            # 可选：用更宽松 flags 再试一次（只对失败的）
            if retry_relaxed:
                try:
                    pyrosetta2, rosetta2, scorefxn2 = init_pyrosetta(flags_relaxed)
                    pose = pose_from_pdb_fixed(pyrosetta2, pdb_file, debug=debug_path)
                    # 用 relaxed 的 scorefxn
                    rosetta, scorefxn = rosetta2, scorefxn2
                except Exception as e2:
                    log_failure("pose_from_pdb_relaxed", row_idx, pdb_str, pdb_id, rec_chains, lig_chains, pdb_file, e2)
                    continue
            else:
                continue

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
            dG_bind=dG_pos,
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
                dG_bind=dG_neg,
            ))

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == total:
            print(f"[Precompute] processed {row_idx+1}/{total} | saved={len(rows_out)} "
                  f"| missing_pdb={skipped_missing_pdb} parse_fail={skipped_parse_fail} rosetta_fail={rosetta_fail}")

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    fail_path = out_csv + ".failures.csv"
    pd.DataFrame(fail_rows).to_csv(fail_path, index=False)

    print(f"[Precompute] DONE. saved_rows={len(out_df)} -> {out_csv}")
    print(f"[Precompute] FAILURES n={len(fail_rows)} -> {fail_path}")
    print(f"[Precompute] missing_pdb={skipped_missing_pdb}, parse_fail={skipped_parse_fail}, rosetta_fail={rosetta_fail}")


# ============================================================
# 6) CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser("Precompute Rosetta binding energies for SKEMPI poses (fixed v2).")
    ap.add_argument("--skempi_csv", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/skempi_v2.csv", help="Path to SKEMPI v2 csv (; separated)") 
    ap.add_argument("--pdb_root", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/SKEMPI v2.0/PDBs", help="Directory containing PDB files (may contain spaces)") 
    ap.add_argument("--out_csv", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/stage3_negative_sample.csv", help="Output CSV path" )

    ap.add_argument("--n_neg_per_pos", type=int, default=5)
    ap.add_argument("--neg_shift_min", type=float, default=20.0)
    ap.add_argument("--neg_shift_max", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--debug_path", action="store_true")
    ap.add_argument("--retry_relaxed", action="store_true",
                    help="Retry failed pdb read with more relaxed flags (optional)")

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
    )


if __name__ == "__main__":
    main()
