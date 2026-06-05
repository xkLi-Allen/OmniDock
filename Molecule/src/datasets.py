# -*- coding: utf-8 -*-
"""
Dataset classes for Stage 2 (surface self-supervised pretraining)
and Stage 3 (protein-ligand supervised fine-tuning).
Self-contained - no imports from other projects.
"""

import os
import json
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ============================================================================
# Shared helpers
# ============================================================================

def _load_npz(path: str, cache: Optional[dict], K: int) -> dict:
    if cache is not None and path in cache:
        return cache[path]

    with np.load(path, allow_pickle=True) as d:
        xs      = d["xs"].astype(np.float32)
        ns      = d["ns"].astype(np.float32)
        centers = d["patch_centers"].astype(np.float32)
        knn     = d["patch_knn_idx"].astype(np.int64)
        order   = d["patch_order"].astype(np.int64)
        meta    = json.loads(str(d["meta"])) if "meta" in d else {}

        def _opt(key, dtype):
            return d[key].astype(dtype) if key in d else None

        lig_pos            = _opt("lig_pos",           np.float32)
        lig_atom_type      = _opt("lig_atom_type",     np.int32)
        lig_atom_element   = _opt("lig_atom_element",  np.int32)
        lig_edge_index     = _opt("lig_edge_index",    np.int32)
        lig_edge_type      = _opt("lig_edge_type",     np.int32)
        lig_center         = _opt("lig_center",        np.float32)
        pocket_label_patch = _opt("pocket_label_patch",np.float32)
        patch_to_lig_dist  = _opt("patch_to_lig_dist", np.float32)

    K0 = knn.shape[1]
    if K0 < K:
        knn = np.concatenate([knn, np.tile(knn[:, -1:], (1, K - K0))], axis=1)
    elif K0 > K:
        knn = knn[:, :K]

    out = dict(
        xs=xs, ns=ns, centers=centers, knn=knn, order=order, meta=meta,
        lig_pos=lig_pos, lig_atom_type=lig_atom_type,
        lig_atom_element=lig_atom_element,
        lig_edge_index=lig_edge_index, lig_edge_type=lig_edge_type,
        lig_center=lig_center,
        pocket_label_patch=pocket_label_patch,
        patch_to_lig_dist=patch_to_lig_dist,
    )
    if cache is not None:
        cache[path] = out
    return out


def _sample_window(arr: dict, seq_len: int, K: int, rng: np.random.Generator):
    xs, ns, centers, knn, order = (
        arr["xs"], arr["ns"], arr["centers"], arr["knn"], arr["order"]
    )
    Nc = centers.shape[0]
    if Nc <= seq_len:
        sel = order
    else:
        start = int(rng.integers(0, Nc - seq_len + 1))
        sel   = order[start : start + seq_len]

    pts_idx = knn[sel]                              # (T, K)
    ctrs    = centers[sel]                          # (T, 3)
    rel_xyz = xs[pts_idx] - ctrs[:, None, :]        # (T, K, 3)
    norms   = ns[pts_idx]                           # (T, K, 3)
    feats   = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
    return feats, rel_xyz.astype(np.float32), ctrs.astype(np.float32), sel.astype(np.int64)


# ============================================================================
# Stage-2 Dataset
# ============================================================================

class SurfacePretrainDataset(Dataset):
    """
    Streams Stage-1 .npz files for self-supervised surface pretraining.

    Item outputs:
        feats   : (T, K, 6)  [rel_xyz | normals]
        coords  : (T, K, 3)  rel_xyz (reconstruction target)
        centers : (T, 3)     patch centre positions
        mask    : (T,)  bool  True = masked patch
    """

    def __init__(self, data_root: str, seq_len: int = 512, K: int = 50,
                 mask_ratio: float = 0.60, cache: bool = True, seed: int = 2024):
        super().__init__()
        self.seq_len    = seq_len
        self.K          = K
        self.mask_ratio = mask_ratio
        self._cache: Optional[dict] = {} if cache else None
        self.files = sorted(
            os.path.join(data_root, f)
            for f in os.listdir(data_root) if f.endswith(".npz")
        )
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {data_root}")
        self._rng_base = np.random.default_rng(seed)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx % len(self.files)]
        arr  = _load_npz(path, self._cache, self.K)
        rng  = np.random.default_rng(self._rng_base.integers(0, 2**31))

        feats, coords, ctrs, _ = _sample_window(arr, self.seq_len, self.K, rng)
        T = feats.shape[0]
        n_mask  = max(1, int(round(self.mask_ratio * T)))
        mask    = np.zeros(T, dtype=np.bool_)
        mask[rng.choice(T, size=n_mask, replace=False)] = True

        return {
            "feats":   torch.from_numpy(feats),
            "coords":  torch.from_numpy(coords),
            "centers": torch.from_numpy(ctrs),
            "mask":    torch.from_numpy(mask),
            "name":    os.path.basename(path),
        }


def pretrain_collate(batch: List[dict]) -> Tuple:
    maxT = max(b["feats"].shape[0] for b in batch)
    K    = batch[0]["feats"].shape[1]
    Xs, Ys, Cs, Ms = [], [], [], []
    for b in batch:
        T = b["feats"].shape[0]; p = maxT - T
        if p > 0:
            Xs.append(torch.cat([b["feats"],   torch.zeros(p, K, 6)], 0))
            Ys.append(torch.cat([b["coords"],  torch.zeros(p, K, 3)], 0))
            Cs.append(torch.cat([b["centers"], torch.zeros(p, 3)],    0))
            Ms.append(torch.cat([b["mask"],    torch.zeros(p, dtype=torch.bool)], 0))
        else:
            Xs.append(b["feats"]); Ys.append(b["coords"])
            Cs.append(b["centers"]); Ms.append(b["mask"])
    return (torch.stack(Xs), torch.stack(Ys), torch.stack(Cs), torch.stack(Ms))
