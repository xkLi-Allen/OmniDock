# -*- coding: utf-8 -*-
"""
Residue geometric neighbourhood for surface points (GeoAN packaging).

For each surface point, find its zeta nearest residue Cα atoms and store:
  - index into the residue list
  - Euclidean distance
  - residue type index

Self-contained - no imports from other projects.
"""

import logging
from typing import Tuple

import numpy as np

from .utils import chunk_indices

logger = logging.getLogger(__name__)


def residue_neighbors(
    surface: np.ndarray,
    ca_pos: np.ndarray,
    ca_type: np.ndarray,
    zeta: int = 16,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each surface point find the zeta nearest Cα atoms.

    Parameters
    ----------
    surface  : (M, 3)  float32  surface point positions
    ca_pos   : (Nr, 3) float32  Cα positions
    ca_type  : (Nr,)   int16    residue type indices
    zeta     : number of neighbours
    chunk    : batch size for distance computation

    Returns
    -------
    nei_idx  : (M, zeta) int32   residue indices (nearest first)
    nei_dist : (M, zeta) float32 distances in Angstroms
    nei_type : (M, zeta) int16   residue type indices
    """
    M  = surface.shape[0]
    Nr = ca_pos.shape[0]
    zeta = min(zeta, Nr)

    nei_idx  = np.empty((M, zeta), dtype=np.int32)
    nei_dist = np.empty((M, zeta), dtype=np.float32)
    nei_type = np.empty((M, zeta), dtype=np.int16)

    for s, e in chunk_indices(M, chunk):
        Sb   = surface[s:e]                                # (b, 3)
        diff = Sb[:, None, :] - ca_pos[None, :, :]        # (b, Nr, 3)
        D    = np.sqrt(np.einsum("bnd,bnd->bn", diff, diff))  # (b, Nr)

        if Nr <= zeta:
            # all residues fit; just sort
            order = np.argsort(D, axis=1)                 # (b, Nr)
            idx_s = order
        else:
            idx_part = np.argpartition(D, zeta - 1, axis=1)[:, :zeta]
            d_part   = np.take_along_axis(D, idx_part, axis=1)
            order    = np.argsort(d_part, axis=1)
            idx_s    = np.take_along_axis(idx_part, order, axis=1)

        d_sorted  = np.take_along_axis(D, idx_s, axis=1)       # (b, zeta)
        t_sorted  = ca_type[idx_s]                              # (b, zeta)

        nei_idx[s:e]  = idx_s.astype(np.int32)
        nei_dist[s:e] = d_sorted.astype(np.float32)
        nei_type[s:e] = t_sorted.astype(np.int16)

    return nei_idx, nei_dist, nei_type
