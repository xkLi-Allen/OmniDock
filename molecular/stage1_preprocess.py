

import os
import sys
import json
import argparse
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.utils          import safe_mkdir, set_seed, is_processed_ok
from src.protein_parser import parse_protein_pdb
from src.ligand_parser  import parse_ligand
from src.surface        import generate_surface
from src.neighbors      import residue_neighbors
from src.patchify       import build_patches
from src.contacts       import compute_pl_contacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset layout helper
# ---------------------------------------------------------------------------

def _find_dataset_dir(dataset_root: str, complex_id: str) -> str:
    """Search sub-year directories for the complex folder."""
    # Direct subfolder: dataset_root/complex_id/
    direct = os.path.join(dataset_root, complex_id)
    if os.path.isdir(direct):
        return direct
    # Year-grouped: dataset_root/YYYY-YYYY/complex_id/
    for sub in os.listdir(dataset_root):
        cand = os.path.join(dataset_root, sub, complex_id)
        if os.path.isdir(cand):
            return cand
    return ""


def _find_ligand_files(cdir: str, cid: str):
    sdf  = os.path.join(cdir, f"{cid}_ligand.sdf")
    mol2 = os.path.join(cdir, f"{cid}_ligand.mol2")
    return (sdf  if os.path.isfile(sdf)  else None,
            mol2 if os.path.isfile(mol2) else None)


# ---------------------------------------------------------------------------
# Per-complex processing
# ---------------------------------------------------------------------------

def process_one(
    complex_id: str,
    dataset_root: str,
    out_dir: str,
    args,
) -> bool:
    out_path = os.path.join(out_dir, f"{complex_id}.npz")
    if not args.overwrite and is_processed_ok(out_path):
        logger.info("[SKIP] %s already processed.", complex_id)
        return True

    cdir = _find_dataset_dir(dataset_root, complex_id)
    if not cdir:
        logger.warning("[SKIP] Directory not found for %s", complex_id)
        return False

    pdb_path = os.path.join(cdir, f"{complex_id}_protein.pdb")
    if not os.path.isfile(pdb_path):
        logger.warning("[SKIP] Protein PDB missing: %s", pdb_path)
        return False

    sdf_path, mol2_path = _find_ligand_files(cdir, complex_id)
    if sdf_path is None and mol2_path is None:
        logger.warning("[SKIP] No ligand file for %s", complex_id)
        return False

    logger.info("[%s] Parsing protein ...", complex_id)
    atom_pos, atom_sigma, ca_pos, ca_type = parse_protein_pdb(pdb_path)

    logger.info("[%s] Generating surface (eta=%d, iters=%d) ...",
                complex_id, args.eta, args.proj_iters)
    xs, ns = generate_surface(
        atom_pos, atom_sigma,
        device       = args.device,
        eta          = args.eta,
        sigma_init   = args.sigma_init,
        r_level      = args.r_level,
        proj_iters   = args.proj_iters,
        proj_lr      = args.proj_lr,
        inner_thresh = args.inner_thresh,
        target_points= args.target_points,
        seed         = args.seed,
    )
    logger.info("[%s] Surface: %d points", complex_id, xs.shape[0])

    logger.info("[%s] Residue neighbours ...", complex_id)
    nei_idx, nei_dist, nei_type = residue_neighbors(
        xs, ca_pos, ca_type, zeta=args.zeta)

    logger.info("[%s] Building patches (fps_ratio=%.3f, K=%d) ...",
                complex_id, args.fps_ratio, args.knn_k)
    fps_idx, patch_centers, patch_knn_idx, patch_morton, patch_order = \
        build_patches(xs, fps_ratio=args.fps_ratio,
                      knn_k=args.knn_k, seed=args.seed)

    logger.info("[%s] Parsing ligand ...", complex_id)
    lig_dict, lig_src = parse_ligand(sdf_path, mol2_path)

    logger.info("[%s] Computing PL contacts (cutoff=%.1f A) ...",
                complex_id, args.pocket_cutoff)
    pl = compute_pl_contacts(
        xs, patch_centers, patch_knn_idx,
        lig_dict["lig_pos"],
        pocket_cutoff=args.pocket_cutoff,
    )

    meta = dict(
        complex_id        = complex_id,
        protein_file      = pdb_path,
        ligand_file       = sdf_path or mol2_path,
        ligand_source_type= lig_src,
        num_surface_points= int(xs.shape[0]),
        num_patch_centers = int(patch_centers.shape[0]),
        num_residues      = int(ca_pos.shape[0]),
        num_lig_atoms     = int(lig_dict["lig_pos"].shape[0]),
        dataset_name      = "PDBbind",
        eta               = args.eta,
        fps_ratio         = args.fps_ratio,
        knn_k             = args.knn_k,
        pocket_cutoff     = args.pocket_cutoff,
    )
    if hasattr(args, "affinity_map") and complex_id in args.affinity_map:
        meta["affinity"] = args.affinity_map[complex_id]

    logger.info("[%s] Saving -> %s", complex_id, out_path)
    save_dict = dict(
        # protein surface
        xs            = xs.astype(np.float32),
        ns            = ns.astype(np.float32),
        ca_pos        = ca_pos.astype(np.float32),
        ca_type       = ca_type.astype(np.int16),
        geo_nei_idx   = nei_idx.astype(np.int32),
        geo_nei_dist  = nei_dist.astype(np.float32),
        geo_nei_type  = nei_type.astype(np.int16),
        # patches
        fps_idx       = fps_idx.astype(np.int64),
        patch_centers = patch_centers.astype(np.float32),
        patch_knn_idx = patch_knn_idx.astype(np.int32),
        patch_morton  = patch_morton,
        patch_order   = patch_order.astype(np.int64),
        # ligand
        lig_pos           = lig_dict["lig_pos"],
        lig_atom_type     = lig_dict["lig_atom_type"],
        lig_atom_element  = lig_dict["lig_atom_element"],
        lig_atom_charge   = lig_dict["lig_atom_charge"],
        lig_atom_aromatic = lig_dict["lig_atom_aromatic"],
        lig_atom_hybrid   = lig_dict["lig_atom_hybrid"],
        lig_atom_numHs    = lig_dict["lig_atom_numHs"],
        lig_edge_index    = lig_dict["lig_edge_index"],
        lig_edge_type     = lig_dict["lig_edge_type"],
        lig_center        = lig_dict["lig_center"],
        # protein-ligand contacts
        surf_to_lig_dist      = pl["surf_to_lig_dist"],
        surf_to_lig_atom_idx  = pl["surf_to_lig_atom_idx"],
        patch_to_lig_dist     = pl["patch_to_lig_dist"],
        pocket_label_point    = pl["pocket_label_point"],
        pocket_label_patch    = pl["pocket_label_patch"],
        patch_contact_score   = pl["patch_contact_score"],
        lig_to_surf_dist      = pl["lig_to_surf_dist"],
        lig_to_patch_idx      = pl["lig_to_patch_idx"],
        # meta
        meta = json.dumps(meta),
    )
    np.savez_compressed(out_path, **save_dict)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Stage 1: Protein-Ligand Surface Preprocessing")
    p.add_argument("--index_file",   required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--out_dir",      default="./outputs/stage1")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    # surface generation
    p.add_argument("--eta",          type=int,   default=8)
    p.add_argument("--sigma_init",   type=float, default=10.0)
    p.add_argument("--r_level",      type=float, default=1.05)
    p.add_argument("--proj_iters",   type=int,   default=100)
    p.add_argument("--proj_lr",      type=float, default=1e-2)
    p.add_argument("--inner_thresh", type=float, default=0.5)
    p.add_argument("--target_points",type=int,   default=20000)
    # patch construction
    p.add_argument("--fps_ratio",    type=float, default=0.05)
    p.add_argument("--knn_k",        type=int,   default=32)
    p.add_argument("--zeta",         type=int,   default=16)
    # protein-ligand
    p.add_argument("--pocket_cutoff",type=float, default=6.0)
    # misc
    p.add_argument("--seed",         type=int,   default=2024)
    p.add_argument("--overwrite",    action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.index_file)
    # build affinity lookup
    if "affinity" in df.columns:
        args.affinity_map = dict(zip(df["complex_id"].astype(str),
                                     df["affinity"]))
    else:
        args.affinity_map = {}

    complexes = df["complex_id"].astype(str).tolist()
    logger.info("Total complexes: %d", len(complexes))

    ok = 0
    for i, cid in enumerate(complexes):
        logger.info("--- [%d/%d] %s ---", i + 1, len(complexes), cid)
        try:
            success = process_one(cid, args.dataset_root, args.out_dir, args)
            if success:
                ok += 1
        except Exception:
            logger.error("[ERROR] %s failed:\n%s", cid, traceback.format_exc())

    logger.info("Done. %d / %d succeeded.", ok, len(complexes))


if __name__ == "__main__":
    main()
