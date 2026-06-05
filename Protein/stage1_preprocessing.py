# -*- coding: utf-8 -*-
"""
Stage 1: Backbone-aware surface preprocessing for SKEMPI per-chain data.

Key improvements over original Stage 1:
  - Extracts and saves backbone N/CA/C coordinates per residue
  - Computes and saves torsion angles (phi, psi, omega) as sin/cos
  - Assigns secondary structure labels from torsion angles
  - Builds surface_patch_to_residue mapping (nearest CA for each patch center)
  - Preserves residue ordering information

Output .npz contains:
  Surface data: xs, ns, patch_centers, patch_knn_idx, patch_order, fps_idx
  Backbone data: backbone_ncac, bb_valid_mask, ca_pos, ca_type, residue_order
  Torsion data: torsion_sincos, torsion_valid_mask
  SS data: ss_labels
  Mapping: surf2res_idx (which residue each patch center is closest to)
"""

import os
import sys
import math
import json
import argparse

import numpy as np
import torch

from Bio.PDB import PDBParser, is_aa

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import (
    AMINO20, RES2IDX,
    compute_backbone_torsions, torsion_to_sincos, assign_ss_from_torsions,
)

# Import surface generation utilities from original code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protein'))
from data_preprocessing import (
    to_tensor, SurfaceSDF, project_to_levelset_gd, sdf_normals,
    remove_inner_points, residue_neighbors, farthest_point_sampling,
    knn_indices, morton3D, safe_mkdir, is_processed_ok, VDW_SIGMA,
)

import pandas as pd


def parse_pdb_chain_group(pdb_path: str, chain_group: str):
    """
    Parse PDB and extract atoms + residue backbone for specified chains.

    Returns:
        atom_pos, atom_sigma: surface generation inputs
        ca_pos: (L, 3) CA positions
        ca_type: (L,) residue type indices
        backbone_ncac: (L, 3, 3) N/CA/C coordinates
        bb_valid: (L,) bool mask for complete backbone
        residue_order: (L,) sequential residue indices (0-based)
    """
    chain_set = set(chain_group)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    atom_pos = []
    atom_sigma = []
    ca_pos = []
    ca_type = []
    backbone_ncac = []
    bb_valid = []
    residue_order = []

    res_idx = 0
    for model in structure:
        for chain in model:
            if chain.id.strip() not in chain_set:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                resname = res.get_resname()
                idx = RES2IDX.get(resname, -1)

                if "CA" not in res:
                    continue

                ca_pos.append(res["CA"].get_coord().astype(np.float32))
                ca_type.append(idx)
                residue_order.append(res_idx)
                res_idx += 1

                has_n = "N" in res
                has_ca = "CA" in res
                has_c = "C" in res
                if has_n and has_ca and has_c:
                    n_coord = res["N"].get_coord().astype(np.float32)
                    ca_coord = res["CA"].get_coord().astype(np.float32)
                    c_coord = res["C"].get_coord().astype(np.float32)
                    backbone_ncac.append([n_coord, ca_coord, c_coord])
                    bb_valid.append(True)
                else:
                    ca_coord = res["CA"].get_coord().astype(np.float32)
                    backbone_ncac.append([ca_coord, ca_coord, ca_coord])
                    bb_valid.append(False)

                for atom in res.get_atoms():
                    elem = atom.element.strip().title()
                    if elem not in VDW_SIGMA:
                        continue
                    atom_pos.append(atom.get_coord().astype(np.float32))
                    atom_sigma.append(VDW_SIGMA[elem])

    if len(atom_pos) == 0:
        raise ValueError(f"No atoms in {pdb_path} for chains '{chain_group}'")

    atom_pos = np.array(atom_pos, dtype=np.float32)
    atom_sigma = np.array(atom_sigma, dtype=np.float32)
    ca_pos = np.array(ca_pos, dtype=np.float32) if ca_pos else np.zeros((0, 3), dtype=np.float32)
    ca_type = np.array(ca_type, dtype=np.int64)
    backbone_ncac = np.array(backbone_ncac, dtype=np.float32) if backbone_ncac else np.zeros((0, 3, 3), dtype=np.float32)
    bb_valid = np.array(bb_valid, dtype=bool)
    residue_order = np.array(residue_order, dtype=np.int64)

    return atom_pos, atom_sigma, ca_pos, ca_type, backbone_ncac, bb_valid, residue_order


def compute_surf2res_mapping(patch_centers: np.ndarray, ca_pos: np.ndarray) -> np.ndarray:
    """Map each patch center to its nearest CA atom (residue index)."""
    if ca_pos.shape[0] == 0:
        return np.zeros(patch_centers.shape[0], dtype=np.int64)
    diff = patch_centers[:, None, :] - ca_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.argmin(axis=1).astype(np.int64)


def process_one_chain(pdb_dir, out_dir, pdb_id, chain_group,
                      device="cuda", eta=8, sigma_init=10.0, r_level=1.05,
                      proj_iters=80, proj_lr=1e-2, inner_thresh=0.5,
                      target_points=10000, fps_ratio=0.05, knn_k=32,
                      zeta=8, seed=2023, overwrite=False):
    """Process one (pdb_id, chain_group) pair into a complete .npz."""
    name = f"{pdb_id}_{chain_group}"
    out_path = os.path.join(out_dir, f"{name}.npz")

    if not overwrite and os.path.isfile(out_path):
        print(f"[SKIP] {out_path}")
        return

    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.isfile(pdb_path):
        print(f"[WARN] PDB not found: {pdb_path}")
        return

    print(f"[{name}] Processing...")

    # 1. Parse PDB
    (atom_pos, atom_sigma, ca_pos, ca_type,
     backbone_ncac, bb_valid, residue_order) = parse_pdb_chain_group(pdb_path, chain_group)

    if ca_pos.shape[0] == 0:
        print(f"[WARN] No residues in {name}")
        return

    # 2. Compute torsion angles
    torsions = compute_backbone_torsions(backbone_ncac, bb_valid)
    torsion_sincos = torsion_to_sincos(torsions)
    torsion_valid = bb_valid.copy()
    # First and last residues have incomplete torsions
    if len(torsion_valid) > 0:
        torsion_valid[0] = False
    if len(torsion_valid) > 1:
        torsion_valid[-1] = False

    # 3. Assign secondary structure
    ss_labels = assign_ss_from_torsions(torsions[:, 0], torsions[:, 1])

    # 4. Generate molecular surface
    A = to_tensor(atom_pos, device)
    SIG = to_tensor(atom_sigma, device)

    centers_np = np.repeat(atom_pos, eta, axis=0)
    noise = np.random.normal(0.0, sigma_init, centers_np.shape).astype(np.float32)
    X0 = to_tensor(centers_np + noise, device)

    sdf = SurfaceSDF(atom_pos, atom_sigma, device=device)
    Xs = project_to_levelset_gd(X0, sdf, r=r_level, iters=proj_iters, lr=proj_lr, momentum=0.0,
                                desc=f"[{name}] Projecting")
    Ns = sdf_normals(Xs, sdf, desc=f"[{name}] Normals")

    Xs_clean, keep = remove_inner_points(Xs, A, SIG, thresh=inner_thresh, desc=f"[{name}] Cleaning")
    Ns_clean = Ns[keep]

    if Xs_clean.shape[0] > target_points:
        Xcpu = Xs_clean.detach().cpu().numpy()
        idx_keep = farthest_point_sampling(Xcpu, target_points, seed=seed)
        Xs_clean = Xs_clean[idx_keep]
        Ns_clean = Ns_clean[idx_keep]

    X_np = Xs_clean.detach().cpu().numpy().astype(np.float32)
    N_np = Ns_clean.detach().cpu().numpy().astype(np.float32)

    # 5. Build patches
    M = X_np.shape[0]
    num_centers = max(1, int(math.ceil(fps_ratio * M)))
    fps_idx = farthest_point_sampling(X_np, num_centers, seed=seed)
    Xc = X_np[fps_idx]
    knn_idx = knn_indices(X_np, Xc, K=knn_k, desc=f"[{name}] KNN")

    # Morton order
    mins = Xc.min(axis=0, keepdims=True)
    maxs = Xc.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    Xc_unit = (Xc - mins) / span
    morton = morton3D(Xc_unit)
    order = np.argsort(morton)

    # 6. Surface-to-residue mapping
    surf2res_idx = compute_surf2res_mapping(Xc, ca_pos)

    # 7. Residue neighbor info for surface points
    nei_idx, nei_dist, nei_type = residue_neighbors(
        X_np, ca_pos.astype(np.float32), ca_type.astype(np.int16), zeta=zeta,
        desc=f"[{name}] Residue NN"
    )

    # 8. Save
    meta = dict(pdb_id=pdb_id, chains=chain_group, n_residues=int(ca_pos.shape[0]),
                n_surface_points=int(M), n_patches=int(num_centers))

    np.savez_compressed(
        out_path,
        # Surface
        xs=X_np, ns=N_np,
        patch_centers=Xc.astype(np.float32),
        patch_knn_idx=knn_idx.astype(np.int32),
        patch_order=order.astype(np.int64),
        fps_idx=fps_idx.astype(np.int64),
        # Backbone
        backbone_ncac=backbone_ncac,
        bb_valid_mask=bb_valid,
        ca_pos=ca_pos,
        ca_type=ca_type.astype(np.int16),
        residue_order=residue_order,
        # Torsion
        torsion_sincos=torsion_sincos,
        torsion_valid_mask=torsion_valid,
        # Secondary structure
        ss_labels=ss_labels,
        # Mapping
        surf2res_idx=surf2res_idx,
        # Residue neighbors
        geo_nei_idx=nei_idx, geo_nei_dist=nei_dist, geo_nei_type=nei_type,
        # Meta
        meta=json.dumps(meta),
    )
    print(f"[{name}] Saved -> {out_path} (residues={ca_pos.shape[0]}, patches={num_centers})")


def parse_pdb_field(pdb_str: str):
    """Parse SKEMPI #Pdb field: '1A4Y_A_B' -> ('1A4Y', 'A', 'B')"""
    parts = str(pdb_str).split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected #Pdb format: {pdb_str}")
    return parts[0], parts[1], parts[2]


def main():
    ap = argparse.ArgumentParser(description="Stage 1: Backbone-aware surface preprocessing")
    ap.add_argument("--skempi_csv", type=str,
                    default="/data/jiangjiaqi/srzhang/InversionDock/Data/Skempi_dataset/skempi_v2.csv")
    ap.add_argument("--pdb_dir", type=str,
                    default="/data/jiangjiaqi/srzhang/InversionDock/Data/Skempi_dataset/Skempiv2")
    ap.add_argument("--out_dir", type=str,
                    default="/data/jiangjiaqi/srzhang/InversionDock/Data/Processed_skempi_backbone_aware")
    ap.add_argument("--eta", type=int, default=8)
    ap.add_argument("--sigma_init", type=float, default=10.0)
    ap.add_argument("--target_points", type=int, default=10000)
    ap.add_argument("--fps_ratio", type=float, default=0.05)
    ap.add_argument("--knn_k", type=int, default=32)
    ap.add_argument("--zeta", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.skempi_csv, sep=";")
    unique_keys = set()
    for _, row in df.iterrows():
        try:
            pdb_id, rec_chains, lig_chains = parse_pdb_field(row["#Pdb"])
        except (ValueError, KeyError):
            continue
        unique_keys.add((pdb_id, rec_chains))
        unique_keys.add((pdb_id, lig_chains))

    print(f"[Stage1] Found {len(unique_keys)} unique (pdb_id, chain_group) pairs")

    for pdb_id, chain_group in tqdm(sorted(unique_keys), desc="Processing"):
        try:
            process_one_chain(
                args.pdb_dir, args.out_dir, pdb_id, chain_group,
                device=args.device, eta=args.eta, sigma_init=args.sigma_init,
                target_points=args.target_points, fps_ratio=args.fps_ratio,
                knn_k=args.knn_k, zeta=args.zeta, seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"[ERROR] {pdb_id}_{chain_group}: {e}")
            continue


if __name__ == "__main__":
    main()
