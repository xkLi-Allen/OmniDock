# -*- coding: utf-8 -*-
"""
Core geometry utilities: tensor helpers, FPS, KNN, Morton encoding.
All functions are self-contained and do NOT import from any other project.
"""

import os
import math
import logging
from typing import Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMINO20 = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
RES2IDX = {r: i for i, r in enumerate(AMINO20)}
IDX2RES = {i: r for r, i in RES2IDX.items()}

# Van der Waals radii (Angstroms) used as Gaussian sigma per element
VDW_SIGMA = {
    "H":  1.20,
    "C":  1.70,
    "N":  1.55,
    "O":  1.52,
    "S":  1.80,
    "SE": 1.90,
    "P":  1.80,
    "F":  1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I":  1.98,
}

# Element -> integer index for ligand atom type encoding
ELEMENT_LIST = ["C", "N", "O", "S", "F", "P", "CL", "BR", "I", "H", "OTHER"]
ELEMENT2IDX = {e: i for i, e in enumerate(ELEMENT_LIST)}


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def to_tensor(x: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to float32 torch tensor on device."""
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalise last dimension."""
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chunk_indices(n: int, chunk: int):
    """Yield (start, end) pairs that partition [0, n) into chunks."""
    for s in range(0, n, chunk):
        yield s, min(n, s + chunk)


def nchunks(n: int, chunk: int) -> int:
    return (n + chunk - 1) // chunk


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Morton (Z-order curve) encoding  - 21 bits per axis -> 63-bit code
# ---------------------------------------------------------------------------

def _part1by2(n: np.ndarray) -> np.ndarray:
    """Spread bits of n so that every third bit position is occupied."""
    n = n.astype(np.uint64)
    n &= np.uint64(0x1fffff)
    n = (n | (n << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    n = (n | (n << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    n = (n | (n << np.uint64(8)))  & np.uint64(0x100f00f00f00f00f)
    n = (n | (n << np.uint64(4)))  & np.uint64(0x10c30c30c30c30c3)
    n = (n | (n << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return n


def morton3D(xyz: np.ndarray) -> np.ndarray:
    """
    Compute Morton code for an (N,3) array of normalised coordinates in [0,1].
    Returns (N,) uint64 Morton codes.
    """
    xyz = np.clip(xyz, 0.0, 1.0)
    q = (xyz * float((1 << 21) - 1)).astype(np.uint64)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    m = _part1by2(x) | (_part1by2(y) << np.uint64(1)) | (_part1by2(z) << np.uint64(2))
    return m.astype(np.uint64)


# ---------------------------------------------------------------------------
# Farthest Point Sampling (pure numpy, O(N*K))
# ---------------------------------------------------------------------------

def farthest_point_sampling(X: np.ndarray, num: int, seed: int = 2024) -> np.ndarray:
    """
    Select `num` points from X (N,3) by greedy farthest-point sampling.
    Returns integer index array of shape (num,).
    """
    N = X.shape[0]
    if num >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = np.empty(num, dtype=np.int64)
    chosen[0] = rng.integers(0, N)
    dists = np.full(N, np.inf, dtype=np.float64)
    for i in range(1, num):
        diff = X - X[chosen[i - 1]][None, :]
        d2 = np.einsum("nd,nd->n", diff, diff)
        np.minimum(dists, d2, out=dists)
        chosen[i] = int(np.argmax(dists))
    return chosen


# ---------------------------------------------------------------------------
# K-Nearest Neighbours (chunked numpy, returns indices sorted by distance)
# ---------------------------------------------------------------------------

def knn_indices(
    X: np.ndarray,
    C: np.ndarray,
    K: int,
    chunk: int = 2048,
) -> np.ndarray:
    """
    For each query center in C (Nc,3), find K nearest points in X (N,3).
    Returns (Nc, K) int64 index array (sorted nearest-first).
    """
    Nc = C.shape[0]
    K = min(K, X.shape[0])
    out = np.empty((Nc, K), dtype=np.int64)
    for s, e in chunk_indices(Nc, chunk):
        Cb = C[s:e]                                       # (b, 3)
        # squared euclidean distance
        diff = Cb[:, None, :] - X[None, :, :]             # (b, N, 3)
        D = np.einsum("bnd,bnd->bn", diff, diff)          # (b, N)
        # partial sort
        idx_part = np.argpartition(D, K - 1, axis=1)[:, :K]
        d_part = np.take_along_axis(D, idx_part, axis=1)
        order = np.argsort(d_part, axis=1)
        out[s:e] = np.take_along_axis(idx_part, order, axis=1)
    return out


# ---------------------------------------------------------------------------
# NPZ validity check
# ---------------------------------------------------------------------------

def is_processed_ok(path: str, required_keys=None) -> bool:
    """
    Return True if the .npz at `path` exists and contains all required_keys
    with non-empty arrays.
    """
    if required_keys is None:
        required_keys = ["xs", "ns", "patch_knn_idx", "patch_order", "meta",
                         "lig_pos", "lig_atom_type"]
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            for k in required_keys:
                if k not in d:
                    return False
            if d["xs"].ndim != 2 or d["xs"].shape[0] == 0:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Random SO(3) rotation matrices
# ---------------------------------------------------------------------------

def random_rotation_matrices(batch_size: int, device) -> torch.Tensor:
    """
    Sample `batch_size` uniformly random rotation matrices from SO(3).
    Uses the Rodrigues formula with random axis and angle.
    Returns (B, 3, 3) tensor.
    """
    axis = torch.randn(batch_size, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    theta = 2 * math.pi * torch.rand(batch_size, device=device)   # (B,)
    ct = torch.cos(theta)   # (B,)
    st = torch.sin(theta)
    vt = 1.0 - ct
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    R = torch.stack([
        torch.stack([ct + kx*kx*vt,   kx*ky*vt - kz*st, kx*kz*vt + ky*st], dim=-1),
        torch.stack([ky*kx*vt + kz*st, ct + ky*ky*vt,   ky*kz*vt - kx*st], dim=-1),
        torch.stack([kz*kx*vt - ky*st, kz*ky*vt + kx*st, ct + kz*kz*vt],  dim=-1),
    ], dim=1)   # (B, 3, 3)
    return R
