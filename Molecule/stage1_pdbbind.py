

import os
import re
import json
import math
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make local imports work regardless of where you run the script from
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Reuse your geometry + IO utilities
from data_preprocessing import (
    to_tensor,
    SurfaceSDF,
    project_to_levelset_gd,
    sdf_normals,
    remove_inner_points,
    residue_neighbors,
    farthest_point_sampling,
    knn_indices,
    morton3D,
    safe_mkdir,
    is_processed_ok,
    parse_pdb_atoms_residues,
)


# ----------------------------
# Ligand parsing via RDKit
# ----------------------------

def try_import_rdkit():
    """Delay-import RDKit so running protein-only doesn't hard-require it."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem  # noqa: F401
        return Chem
    except Exception:
        return None


def parse_ligand_atoms_rdkit(lig_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ligand.sdf/mol2 -> (atom_pos, atom_sigma)

    Returns:
      atom_pos: (Na,3) float32
      atom_sigma: (Na,) float32 (vdW radius in Å)

    Notes:
      - Uses RDKit PeriodicTable.GetRvdw for vdW radii; falls back to 1.7Å.
      - Requires a 3D conformer.
    """
    Chem = try_import_rdkit()
    if Chem is None:
        raise ImportError("RDKit not found. Install RDKit to parse ligand .sdf/.mol2")

    ext = os.path.splitext(lig_path)[1].lower()
    mol = None

    if ext == ".sdf":
        supplier = Chem.SDMolSupplier(lig_path, removeHs=False, sanitize=False)
        for m in supplier:
            if m is not None:
                mol = m
                break
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(lig_path, removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand extension: {ext}")

    if mol is None:
        raise ValueError(f"Failed to read ligand file: {lig_path}")

    # Best-effort sanitize (don't fail the whole sample)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    if mol.GetNumConformers() == 0:
        raise ValueError(f"No 3D conformer found in ligand file: {lig_path}")

    conf = mol.GetConformer(0)
    pt = Chem.GetPeriodicTable()

    atom_pos: List[np.ndarray] = []
    atom_sigma: List[float] = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        p = conf.GetAtomPosition(idx)
        atom_pos.append(np.array([p.x, p.y, p.z], dtype=np.float32))

        Z = atom.GetAtomicNum()
        try:
            rvdw = float(pt.GetRvdw(Z))
        except Exception:
            rvdw = 0.0
        if rvdw <= 0:
            rvdw = 1.7
        atom_sigma.append(rvdw)

    atom_pos_np = np.asarray(atom_pos, dtype=np.float32)
    atom_sigma_np = np.asarray(atom_sigma, dtype=np.float32)

    if atom_pos_np.shape[0] == 0:
        raise ValueError(f"No atoms parsed from ligand file: {lig_path}")

    return atom_pos_np, atom_sigma_np


# ----------------------------
# Scanning / resolving files
# ----------------------------

def read_pdbbind_index(index_file: str) -> List[str]:
    """Read a PDBBind INDEX_*.lst file and return pdb ids (lowercase)."""
    pdb_ids: List[str] = []
    with open(index_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()[0].strip()
            if re.fullmatch(r"[0-9a-zA-Z]{4}", tok):
                pdb_ids.append(tok.lower())
    return pdb_ids


def scan_pdbbind_subset(pdbbind_root: str, subset: str) -> Dict[str, str]:
    """Scan pdbbind_root/subset and return {pdb_id -> abs_dir}."""
    subset_dir = os.path.join(pdbbind_root, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    items: Dict[str, str] = {}
    for name in sorted(os.listdir(subset_dir)):
        d = os.path.join(subset_dir, name)
        if os.path.isdir(d) and re.fullmatch(r"[0-9a-zA-Z]{4}", name):
            items[name.lower()] = d
    return items


def resolve_pdbbind_files(sample_dir: str, pdb_id: str) -> Dict[str, Optional[str]]:
    """Resolve input files for a complex directory.

    Supports both naming styles:
      - receptor: {pid}_receptor.pdb (preferred) or {pid}_protein.pdb
      - pocket:   {pid}_pocket.pdb
      - ligand:   {pid}_ligand.sdf or {pid}_ligand.mol2

    Returns dict keys: protein, pocket, ligand (protein == receptor).
    """
    pid = pdb_id.lower()

    receptor = os.path.join(sample_dir, f"{pid}_receptor.pdb")
    protein  = os.path.join(sample_dir, f"{pid}_protein.pdb")
    pocket   = os.path.join(sample_dir, f"{pid}_pocket.pdb")
    sdf      = os.path.join(sample_dir, f"{pid}_ligand.sdf")
    mol2     = os.path.join(sample_dir, f"{pid}_ligand.mol2")

    prot_path = receptor if os.path.isfile(receptor) else (protein if os.path.isfile(protein) else None)

    lig_path = None
    if os.path.isfile(sdf):
        lig_path = sdf
    elif os.path.isfile(mol2):
        lig_path = mol2

    return {
        "protein": prot_path,
        "pocket": pocket if os.path.isfile(pocket) else None,
        "ligand": lig_path,
    }


# ----------------------------
# Core: build payload
# ----------------------------

@dataclass
class Task:
    pdb_id: str
    kind: str          # protein / pocket / ligand
    in_path: str
    out_path: str


def _pad_residue_neighbors(
    nei_idx: np.ndarray,
    nei_dist: np.ndarray,
    nei_type: np.ndarray,
    zeta_target: int,
    pad_idx: int = -1,
    pad_type: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad residue neighbor arrays from (M,zeta_eff) to (M,zeta_target)."""
    M, zeta_eff = nei_idx.shape
    if zeta_eff == zeta_target:
        return nei_idx, nei_dist, nei_type

    out_idx = np.full((M, zeta_target), pad_idx, dtype=np.int32)
    out_dist = np.full((M, zeta_target), np.inf, dtype=np.float32)
    out_type = np.full((M, zeta_target), pad_type, dtype=np.int16)

    out_idx[:, :zeta_eff] = nei_idx.astype(np.int32)
    out_dist[:, :zeta_eff] = nei_dist.astype(np.float32)
    out_type[:, :zeta_eff] = nei_type.astype(np.int16)
    return out_idx, out_dist, out_type


def _safe_residue_neighbors(
    X_np: np.ndarray,
    ca_pos_np: np.ndarray,
    ca_type_np: np.ndarray,
    zeta: int,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call residue_neighbors safely when Nr < zeta, with padding."""
    M = X_np.shape[0]
    R = ca_pos_np.shape[0]
    if M == 0 or R == 0:
        return (
            np.full((M, zeta), -1, dtype=np.int32),
            np.full((M, zeta), np.inf, dtype=np.float32),
            np.full((M, zeta), -1, dtype=np.int16),
        )

    zeta_eff = min(zeta, int(R))
    zeta_eff = max(1, zeta_eff)

    nei_idx, nei_dist, nei_type = residue_neighbors(
        X_np,
        ca_pos_np.astype(np.float32),
        ca_type_np.astype(np.int16),
        zeta=zeta_eff,
        desc=desc,
    )
    return _pad_residue_neighbors(nei_idx, nei_dist, nei_type, zeta_target=zeta)


def _safe_knn_indices(
    X_np: np.ndarray,
    Xc: np.ndarray,
    knn_k: int,
    desc: str,
) -> np.ndarray:
    """Call knn_indices safely when M < knn_k, with padding by repeating last."""
    M = X_np.shape[0]
    Nc = Xc.shape[0]
    if M == 0 or Nc == 0:
        return np.zeros((Nc, knn_k), dtype=np.int64)

    K_eff = min(knn_k, int(M))
    K_eff = max(1, K_eff)
    knn_eff = knn_indices(X_np, Xc, K=K_eff, desc=desc).astype(np.int64)  # (Nc,K_eff)

    if K_eff == knn_k:
        return knn_eff

    pad_val = knn_eff[:, -1:]
    out = np.repeat(pad_val, repeats=knn_k, axis=1)
    out[:, :K_eff] = knn_eff
    return out


def _morton_order_for_centers(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute morton codes and order for centers. Requires normalization to [0,1]."""
    if Xc.shape[0] == 0:
        return np.zeros((0,), dtype=np.uint64), np.zeros((0,), dtype=np.int64)

    mins = Xc.min(axis=0)
    maxs = Xc.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    Xc_unit = (Xc - mins) / span
    Xc_unit = np.clip(Xc_unit, 0.0, 1.0).astype(np.float32)

    morton = morton3D(Xc_unit)
    order = np.argsort(morton).astype(np.int64)
    return morton, order


def build_surface_payload(
    name: str,
    atom_pos_np: np.ndarray,
    atom_sig_np: np.ndarray,
    ca_pos_np: Optional[np.ndarray],
    ca_type_np: Optional[np.ndarray],
    device: str,
    eta: int,
    sigma_init: float,
    r_level: float,
    proj_iters: int,
    proj_lr: float,
    inner_thresh: float,
    target_points: int,
    fps_ratio: float,
    knn_k: int,
    zeta: int,
    seed: int,
    has_residue: bool,
) -> Dict[str, np.ndarray]:
    """Surface-VQMAE Stage-1 pipeline to produce a np.savez payload."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # torch buffers (remove_inner_points expects atom_sigma shape (Na,))
    A = to_tensor(atom_pos_np, device=device)                 # (Na,3)
    SIG = to_tensor(atom_sig_np, device=device)              # (Na,)

    # 1) Sampling around atoms
    centers = np.repeat(atom_pos_np, eta, axis=0)
    noise = np.random.normal(0.0, sigma_init, size=centers.shape).astype(np.float32)
    X0_np = centers + noise
    X0 = to_tensor(X0_np, device=device)

    # 2) Project to SDF level-set
    sdf = SurfaceSDF(atom_pos_np, atom_sig_np, device=device)
    Xs = project_to_levelset_gd(
        X0, sdf, r=r_level, iters=proj_iters, lr=proj_lr, momentum=0.0,
        desc=f"[{name}] Projecting",
    )

    # 3) Normals
    Ns = sdf_normals(Xs, sdf, desc=f"[{name}] Normals")

    # 4) Remove inner points + optional FPS downsample
    Xs_clean, keep = remove_inner_points(
        Xs, A, SIG, thresh=inner_thresh, desc=f"[{name}] Cleaning"
    )
    Ns_clean = Ns[keep]

    if Xs_clean.shape[0] == 0:
        raise ValueError(f"[{name}] No surface points left after cleaning")

    if Xs_clean.shape[0] > target_points:
        Xcpu = Xs_clean.detach().cpu().numpy()
        idx_keep = farthest_point_sampling(Xcpu, target_points, seed=seed)
        Xs_clean = Xs_clean[idx_keep]
        Ns_clean = Ns_clean[idx_keep]

    X_np = Xs_clean.detach().cpu().numpy().astype(np.float32)  # (M,3)
    N_np = Ns_clean.detach().cpu().numpy().astype(np.float32)  # (M,3)
    M = int(X_np.shape[0])

    # 5) GeoAN residue neighbors
    if has_residue and ca_pos_np is not None and ca_pos_np.shape[0] > 0:
        nei_idx, nei_dist, nei_type = _safe_residue_neighbors(
            X_np,
            ca_pos_np.astype(np.float32),
            ca_type_np.astype(np.int16),
            zeta=zeta,
            desc=f"[{name}] Residue NN",
        )
    else:
        nei_idx = np.full((M, zeta), -1, dtype=np.int32)
        nei_dist = np.full((M, zeta), np.inf, dtype=np.float32)
        nei_type = np.full((M, zeta), -1, dtype=np.int16)
        ca_pos_np = np.zeros((0, 3), dtype=np.float32)
        ca_type_np = np.zeros((0,), dtype=np.int64)

    # 6) FPS + KNN patches
    num_centers = max(1, int(math.ceil(fps_ratio * M)))
    fps_idx = farthest_point_sampling(X_np, num_centers, seed=seed).astype(np.int64)
    Xc = X_np[fps_idx].astype(np.float32)  # (Nc,3)

    knn_idx = _safe_knn_indices(X_np, Xc, knn_k=knn_k, desc=f"[{name}] KNN").astype(np.int64)

    # 7) Morton sorting (normalize centers first!)
    morton, order = _morton_order_for_centers(Xc)

    payload = dict(
        xs=X_np,
        ns=N_np,
        ca_pos=ca_pos_np.astype(np.float32),
        ca_type=ca_type_np.astype(np.int64),
        geo_nei_idx=nei_idx.astype(np.int32),
        geo_nei_dist=nei_dist.astype(np.float32),
        geo_nei_type=nei_type.astype(np.int16),
        patch_centers=Xc.astype(np.float32),
        patch_knn_idx=knn_idx.astype(np.int32),
        patch_morton=morton,
        patch_order=order,
        fps_idx=fps_idx,
    )
    return payload


def _atomic_save_npz(out_path: str, **kwargs):
    """Write npz atomically to avoid half-written files."""
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, **kwargs)
    os.replace(tmp, out_path)


def run_task(
    task: Task,
    device: str,
    eta: int,
    sigma_init: float,
    r_level: float,
    proj_iters: int,
    proj_lr: float,
    inner_thresh: float,
    target_points: int,
    fps_ratio: float,
    knn_k: int,
    zeta: int,
    seed: int,
    overwrite: bool,
):
    """Execute one task: read structure -> build payload -> save .npz"""

    if (not overwrite) and os.path.isfile(task.out_path) and is_processed_ok(task.out_path):
        print(f"[SKIP] {task.kind} {task.pdb_id} (exists)")
        return

    safe_mkdir(os.path.dirname(task.out_path))

    name = f"{task.pdb_id}_{task.kind}"

    # parse structure
    if task.kind in ("protein", "pocket"):
        atom_pos_np, atom_sig_np, ca_pos_np, ca_type_np = parse_pdb_atoms_residues(task.in_path)
        has_residue = True
    elif task.kind == "ligand":
        atom_pos_np, atom_sig_np = parse_ligand_atoms_rdkit(task.in_path)
        ca_pos_np, ca_type_np = None, None
        has_residue = False
    else:
        raise ValueError(f"Unknown kind: {task.kind}")

    payload = build_surface_payload(
        name=name,
        atom_pos_np=atom_pos_np,
        atom_sig_np=atom_sig_np,
        ca_pos_np=ca_pos_np,
        ca_type_np=ca_type_np,
        device=device,
        eta=eta,
        sigma_init=sigma_init,
        r_level=r_level,
        proj_iters=proj_iters,
        proj_lr=proj_lr,
        inner_thresh=inner_thresh,
        target_points=target_points,
        fps_ratio=fps_ratio,
        knn_k=knn_k,
        zeta=zeta,
        seed=seed,
        has_residue=has_residue,
    )

    meta = dict(
        pdb_id=task.pdb_id,
        kind=task.kind,
        in_path=task.in_path,
        out_path=task.out_path,
        device=device,
        eta=eta,
        sigma_init=sigma_init,
        r_level=r_level,
        proj_iters=proj_iters,
        proj_lr=proj_lr,
        inner_thresh=inner_thresh,
        target_points=target_points,
        fps_ratio=fps_ratio,
        knn_k=knn_k,
        zeta=zeta,
        seed=seed,
        has_residue=has_residue,
    )

    _atomic_save_npz(task.out_path, **payload, meta=json.dumps(meta))
    print(f"[OK] {task.kind} {task.pdb_id} -> {task.out_path}")


# ----------------------------
# Task building
# ----------------------------

def build_tasks(
    pdbbind_root: str,
    subset: str,
    out_dir: str,
    kinds: List[str],
    index_file: Optional[str],
    require_files: bool,
) -> List[Task]:
    """Scan subset and build tasks."""
    all_items = scan_pdbbind_subset(pdbbind_root, subset)

    allowed = None
    if index_file is not None:
        allowed = set(read_pdbbind_index(index_file))

    tasks: List[Task] = []
    for pdb_id, sample_dir in all_items.items():
        if allowed is not None and pdb_id not in allowed:
            continue

        paths = resolve_pdbbind_files(sample_dir, pdb_id)

        for k in kinds:
            in_path = paths.get(k)
            if in_path is None:
                if require_files:
                    continue
                else:
                    in_path = ""

            out_path = os.path.join(out_dir, k, f"{pdb_id}.npz")
            tasks.append(Task(pdb_id=pdb_id, kind=k, in_path=in_path, out_path=out_path))

    return tasks


def main():
    parser = argparse.ArgumentParser("Stage-1 PDBBind surface preprocessing")

    parser.add_argument("--pdbbind_root", type=str, required=True,
                        help="PDBBind root directory (contains refined-set/ or general-set/)")
    parser.add_argument("--subset", type=str, default="refined-set",
                        choices=["refined-set", "general-set"],
                        help="Subset to process")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory (writes out_dir/{kind}/{pdb_id}.npz)")

    parser.add_argument("--kinds", type=str, nargs="+", default=["pocket", "ligand"],
                        choices=["protein", "pocket", "ligand"],
                        help="Kinds to process. NOTE: 'protein' means receptor.")

    parser.add_argument("--index_file", type=str, default=None,
                        help="Optional: PDBBind INDEX_*.lst to filter pdb ids")
    parser.add_argument("--require_files", action="store_true",
                        help="If set: skip samples missing required input files")

    # Stage-1 hyperparameters
    parser.add_argument("--eta", type=int, default=20)
    parser.add_argument("--sigma_init", type=float, default=10.0)
    parser.add_argument("--r_level", type=float, default=1.05)
    parser.add_argument("--proj_iters", type=int, default=200)
    parser.add_argument("--proj_lr", type=float, default=1e-2)
    parser.add_argument("--inner_thresh", type=float, default=0.5)
    parser.add_argument("--target_points", type=int, default=50000)
    parser.add_argument("--fps_ratio", type=float, default=0.05)
    parser.add_argument("--knn_k", type=int, default=50)
    parser.add_argument("--zeta", type=int, default=16)

    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    tasks = build_tasks(
        pdbbind_root=args.pdbbind_root,
        subset=args.subset,
        out_dir=args.out_dir,
        kinds=args.kinds,
        index_file=args.index_file,
        require_files=args.require_files,
    )

    print(f"[INFO] total tasks: {len(tasks)}")

    # Serial execution (stable). You can parallelize later if needed.
    for t in tasks:
        if not t.in_path:
            print(f"[WARN] Missing input for {t.kind} {t.pdb_id}, skipped (no in_path).")
            continue
        try:
            run_task(
                task=t,
                device=args.device,
                eta=args.eta,
                sigma_init=args.sigma_init,
                r_level=args.r_level,
                proj_iters=args.proj_iters,
                proj_lr=args.proj_lr,
                inner_thresh=args.inner_thresh,
                target_points=args.target_points,
                fps_ratio=args.fps_ratio,
                knn_k=args.knn_k,
                zeta=args.zeta,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"[FAIL] {t.kind} {t.pdb_id}: {e}")


if __name__ == "__main__":
    main()