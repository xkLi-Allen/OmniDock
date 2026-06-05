# -*- coding: utf-8 -*-
"""
Protein-ligand contact / proximity labels.

Computes:
  surf_to_lig_dist      : (M,)    min distance from each surface point to any ligand atom
  surf_to_lig_atom_idx  : (M,)    index of the nearest ligand atom per surface point
  patch_to_lig_dist     : (Nc,)   min distance from each patch centre to any ligand atom
  pocket_label_point    : (M,)    bool  surface point is within pocket_cutoff of any ligand atom
  pocket_label_patch    : (Nc,)   bool  patch centre is within pocket_cutoff
  patch_contact_score   : (Nc,)   float32  fraction of KNN points that are within pocket_cutoff
  lig_to_surf_dist      : (Na,)   min distance from each ligand atom to any surface point
  lig_to_patch_idx      : (Na,)   index of the nearest patch centre per ligand atom

Self-contained - no imports from other projects.
"""

import logging
from typing import Tuple, Dict

import numpy as np

from .utils import chunk_indices

logger = logging.getLogger(__name__)


def _pairwise_min_dist(
    A: np.ndarray,   # (M, 3)
    B: np.ndarray,   # (N, 3)
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row in A compute the minimum L2 distance to any row in B,
    and the index of that row in B.

    Returns
    -------
    min_dist : (M,) float32
    min_idx  : (M,) int32
    """
    M = A.shape[0]
    min_dist = np.full(M, np.inf, dtype=np.float64)
    min_idx  = np.zeros(M, dtype=np.int32)

    for s, e in chunk_indices(M, chunk):
        Ab = A[s:e]                                          # (b, 3)
        diff = Ab[:, None, :] - B[None, :, :]               # (b, N, 3)
        D    = np.sqrt(np.einsum("bnd,bnd->bn", diff, diff)) # (b, N)
        idx  = D.argmin(axis=1)
        d    = D[np.arange(e - s), idx]
        min_dist[s:e] = d
        min_idx[s:e]  = idx.astype(np.int32)

    return min_dist.astype(np.float32), min_idx


def compute_pl_contacts(
    surf_pts: np.ndarray,       # (M, 3)  surface points
    patch_centers: np.ndarray,  # (Nc, 3) patch centres
    patch_knn_idx: np.ndarray,  # (Nc, K) KNN indices into surf_pts
    lig_pos: np.ndarray,        # (Na, 3) ligand atom positions
    pocket_cutoff: float = 6.0,
) -> Dict[str, np.ndarray]:
    """
    Compute all protein-ligand proximity / contact labels needed for Stage 1 NPZ.

    Parameters
    ----------
    surf_pts      : (M,  3) surface point positions
    patch_centers : (Nc, 3) patch centre positions
    patch_knn_idx : (Nc, K) int32, indices into surf_pts for each patch
    lig_pos       : (Na, 3) ligand atom positions
    pocket_cutoff : distance threshold (Angstroms) for pocket labels

    Returns
    -------
    dict with keys:
        surf_to_lig_dist      (M,)   float32
        surf_to_lig_atom_idx  (M,)   int32
        patch_to_lig_dist     (Nc,)  float32
        pocket_label_point    (M,)   bool
        pocket_label_patch    (Nc,)  bool
        patch_contact_score   (Nc,)  float32
        lig_to_surf_dist      (Na,)  float32
        lig_to_patch_idx      (Na,)  int32
    """
    if lig_pos.shape[0] == 0:
        M  = surf_pts.shape[0]
        Nc = patch_centers.shape[0]
        Na = 0
        return dict(
            surf_to_lig_dist      = np.full(M,  999.0, dtype=np.float32),
            surf_to_lig_atom_idx  = np.zeros(M,        dtype=np.int32),
            patch_to_lig_dist     = np.full(Nc, 999.0, dtype=np.float32),
            pocket_label_point    = np.zeros(M,        dtype=np.bool_),
            pocket_label_patch    = np.zeros(Nc,       dtype=np.bool_),
            patch_contact_score   = np.zeros(Nc,       dtype=np.float32),
            lig_to_surf_dist      = np.empty(0,        dtype=np.float32),
            lig_to_patch_idx      = np.empty(0,        dtype=np.int32),
        )

    # Surface pts -> ligand atoms
    surf_to_lig_dist, surf_to_lig_atom_idx = _pairwise_min_dist(surf_pts, lig_pos)
    pocket_label_point = surf_to_lig_dist <= pocket_cutoff

    # Patch centres -> ligand atoms
    patch_to_lig_dist, _ = _pairwise_min_dist(patch_centers, lig_pos)
    pocket_label_patch   = patch_to_lig_dist <= pocket_cutoff

    # Patch contact score: fraction of KNN points within pocket_cutoff
    Nc, K = patch_knn_idx.shape
    knn_dists = surf_to_lig_dist[patch_knn_idx.ravel()].reshape(Nc, K)  # (Nc, K)
    patch_contact_score = (knn_dists <= pocket_cutoff).mean(axis=1).astype(np.float32)

    # Ligand atoms -> surface pts
    lig_to_surf_dist, _ = _pairwise_min_dist(lig_pos, surf_pts)

    # Ligand atoms -> patch centres
    lig_to_patch_dist, lig_to_patch_idx = _pairwise_min_dist(lig_pos, patch_centers)

    return dict(
        surf_to_lig_dist      = surf_to_lig_dist,
        surf_to_lig_atom_idx  = surf_to_lig_atom_idx,
        patch_to_lig_dist     = patch_to_lig_dist,
        pocket_label_point    = pocket_label_point,
        pocket_label_patch    = pocket_label_patch,
        patch_contact_score   = patch_contact_score,
        lig_to_surf_dist      = lig_to_surf_dist,
        lig_to_patch_idx      = lig_to_patch_idx,
    )
