# -*- coding: utf-8 -*-
"""
Patch construction for surface point clouds:
  1. FPS to select patch centres
  2. KNN to assign K surface points to each centre
  3. Morton code ordering of patch centres

Self-contained - no imports from other projects.
"""

import math
import logging
from typing import Tuple

import numpy as np

from .utils import farthest_point_sampling, knn_indices, morton3D

logger = logging.getLogger(__name__)


def build_patches(
    X: np.ndarray,
    fps_ratio: float = 0.05,
    knn_k: int = 50,
    seed: int = 2024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build surface patches from a point cloud.

    Parameters
    ----------
    X         : (M, 3) float32  surface points
    fps_ratio : fraction of M to use as patch centres
    knn_k     : number of nearest surface points per patch
    seed      : random seed for FPS

    Returns
    -------
    fps_idx      : (Nc,)      int64  indices into X for patch centres
    patch_centers: (Nc, 3)   float32 Cartesian coords of patch centres
    patch_knn_idx: (Nc, K)   int32  indices into X for each patch's K points
    patch_morton : (Nc,)     uint64 Morton codes of patch centres
    patch_order  : (Nc,)     int64  argsort of Morton codes (spatial order)
    """
    M  = X.shape[0]
    Nc = max(1, math.ceil(fps_ratio * M))

    # FPS
    fps_idx = farthest_point_sampling(X, Nc, seed=seed)
    Xc      = X[fps_idx]                                  # (Nc, 3)

    # KNN: for each patch centre find K nearest surface points
    K = min(knn_k, M)
    patch_knn_idx = knn_indices(X, Xc, K=K)              # (Nc, K)

    # Morton ordering
    mins = Xc.min(axis=0, keepdims=True)
    maxs = Xc.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    Xc_unit = (Xc - mins) / span                         # normalise to [0,1]
    patch_morton = morton3D(Xc_unit)                     # (Nc,) uint64
    patch_order  = np.argsort(patch_morton).astype(np.int64)  # (Nc,)

    return (
        fps_idx.astype(np.int64),
        Xc.astype(np.float32),
        patch_knn_idx.astype(np.int32),
        patch_morton,
        patch_order,
    )
