# -*- coding: utf-8 -*-
"""
Stage-3 protein-ligand supervised dataset.
Imported by stage3_finetune.py directly.
"""

import os
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _load_npz_s3(path: str, cache: Optional[dict], K: int) -> dict:
    if cache is not None and path in cache:
        return cache[path]
    with np.load(path, allow_pickle=True) as d:
        xs      = d["xs"].astype(np.float32)
        ns      = d["ns"].astype(np.float32)
        centers = d["patch_centers"].astype(np.float32)
        knn     = d["patch_knn_idx"].astype(np.int64)
        order   = d["patch_order"].astype(np.int64)
        meta    = json.loads(str(d["meta"])) if "meta" in d else {}
        def _f(key, dtype):
            return d[key].astype(dtype) if key in d else None
        lig_pos            = _f("lig_pos",           np.float32)
        lig_atom_type      = _f("lig_atom_type",     np.int32)
        lig_edge_index     = _f("lig_edge_index",    np.int32)
        lig_edge_type      = _f("lig_edge_type",     np.int32)
        lig_center         = _f("lig_center",        np.float32)
        pocket_label_patch = _f("pocket_label_patch",np.float32)
        patch_to_lig_dist  = _f("patch_to_lig_dist", np.float32)
    K0 = knn.shape[1]
    if K0 < K:
        knn = np.concatenate([knn, np.tile(knn[:, -1:], (1, K - K0))], axis=1)
    elif K0 > K:
        knn = knn[:, :K]
    out = dict(
        xs=xs, ns=ns, centers=centers, knn=knn, order=order, meta=meta,
        lig_pos=lig_pos, lig_atom_type=lig_atom_type,
        lig_edge_index=lig_edge_index, lig_edge_type=lig_edge_type,
        lig_center=lig_center,
        pocket_label_patch=pocket_label_patch,
        patch_to_lig_dist=patch_to_lig_dist,
    )
    if cache is not None:
        cache[path] = out
    return out


def _sample_rec_window(arr: dict, seq_len: int, K: int,
                       rng: np.random.Generator) -> Tuple:
    xs, ns, centers, knn, order = (
        arr["xs"], arr["ns"], arr["centers"], arr["knn"], arr["order"]
    )
    Nc = centers.shape[0]
    if Nc <= seq_len:
        sel = order
    else:
        start = int(rng.integers(0, Nc - seq_len + 1))
        sel   = order[start: start + seq_len]
    pts_idx = knn[sel]
    ctrs    = centers[sel]
    rel_xyz = xs[pts_idx] - ctrs[:, None, :]
    norms   = ns[pts_idx]
    feats   = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
    plp     = arr["pocket_label_patch"]
    pocket_sel = plp[sel].astype(np.float32) if plp is not None \
        else np.zeros(len(sel), dtype=np.float32)
    return feats, ctrs.astype(np.float32), pocket_sel, sel.astype(np.int64)


class ProteinLigandDataset(Dataset):
    """
    Supervised protein-ligand dataset for Stage-3.

    index_file CSV: complex_id, affinity, split
    npz_root: {complex_id}.npz Stage-1 outputs
    """

    def __init__(self, index_file: str, npz_root: str,
                 seq_len: int = 512, K: int = 50,
                 split: Optional[str] = None,
                 cache: bool = True, seed: int = 2024):
        super().__init__()
        self.seq_len   = seq_len
        self.K         = K
        self._cache: Optional[dict] = {} if cache else None
        self._rng_base = np.random.default_rng(seed)

        df = pd.read_csv(index_file)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.samples: List[dict] = []
        missing = 0
        for _, row in df.iterrows():
            cid = str(row["complex_id"])
            npz = os.path.join(npz_root, f"{cid}.npz")
            if not os.path.isfile(npz):
                missing += 1
                continue
            aff = float(row["affinity"]) \
                if "affinity" in row and not pd.isna(row["affinity"]) else 0.0
            self.samples.append(dict(complex_id=cid, npz=npz, affinity=aff))

        logger.info("ProteinLigandDataset: %d loaded, %d missing npz",
                    len(self.samples), missing)
        if not self.samples:
            raise RuntimeError(
                f"No samples found (missing={missing}). Check npz_root={npz_root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]
        arr  = _load_npz_s3(info["npz"], self._cache, self.K)
        rng  = np.random.default_rng(self._rng_base.integers(0, 2**31))

        rec_feats, rec_centers, pocket_labels, _ = _sample_rec_window(
            arr, self.seq_len, self.K, rng)

        def _lig(key, fb):
            v = arr[key]
            return v if v is not None else fb

        lig_pos        = _lig("lig_pos",        np.zeros((1, 3), dtype=np.float32))
        lig_atom_type  = _lig("lig_atom_type",  np.zeros(1,      dtype=np.int32))
        lig_edge_index = _lig("lig_edge_index", np.zeros((2, 0), dtype=np.int32))
        lig_edge_type  = _lig("lig_edge_type",  np.zeros(0,      dtype=np.int32))
        lig_center     = _lig("lig_center",     np.zeros(3,      dtype=np.float32))

        return {
            "rec_feats":     torch.from_numpy(rec_feats),
            "rec_centers":   torch.from_numpy(rec_centers),
            "pocket_labels": torch.from_numpy(pocket_labels),
            "lig_pos":       torch.from_numpy(lig_pos),
            "lig_atom_type": torch.from_numpy(lig_atom_type).long(),
            "lig_edge_index":torch.from_numpy(lig_edge_index),
            "lig_edge_type": torch.from_numpy(lig_edge_type).long(),
            "lig_center":    torch.from_numpy(lig_center),
            "affinity":      torch.tensor([info["affinity"]], dtype=torch.float32),
            "bind_label":    torch.tensor([1.0], dtype=torch.float32),
            "name":          info["complex_id"],
        }


def finetune_collate(batch: List[dict]) -> dict:
    """Collate with padding for variable-length receptor / ligand sequences."""
    K      = batch[0]["rec_feats"].shape[1]
    Tr_max = max(b["rec_feats"].shape[0] for b in batch)
    Na_max = max(b["lig_pos"].shape[0]   for b in batch)

    rec_feats_l, rec_centers_l, pocket_l, rec_mask_l = [], [], [], []
    lig_pos_l, lig_type_l, lig_mask_l = [], [], []
    lig_ctr_l, aff_l, bind_l = [], [], []
    ei_list, et_list = [], []
    names = []

    for b in batch:
        Tr = b["rec_feats"].shape[0]; pr = Tr_max - Tr
        if pr > 0:
            rec_feats_l.append(torch.cat([b["rec_feats"],   torch.zeros(pr, K, 6)], 0))
            rec_centers_l.append(torch.cat([b["rec_centers"],torch.zeros(pr, 3)],   0))
            pocket_l.append(torch.cat([b["pocket_labels"],  torch.zeros(pr)],       0))
            rec_mask_l.append(torch.cat([
                torch.zeros(Tr, dtype=torch.bool),
                torch.ones( pr, dtype=torch.bool)], 0))
        else:
            rec_feats_l.append(b["rec_feats"])
            rec_centers_l.append(b["rec_centers"])
            pocket_l.append(b["pocket_labels"])
            rec_mask_l.append(torch.zeros(Tr, dtype=torch.bool))

        Na = b["lig_pos"].shape[0]; pl = Na_max - Na
        if pl > 0:
            lig_pos_l.append(torch.cat([b["lig_pos"],  torch.zeros(pl, 3)],          0))
            lig_type_l.append(torch.cat([b["lig_atom_type"],
                                         torch.zeros(pl, dtype=torch.long)],          0))
            lig_mask_l.append(torch.cat([
                torch.zeros(Na, dtype=torch.bool),
                torch.ones( pl, dtype=torch.bool)], 0))
        else:
            lig_pos_l.append(b["lig_pos"])
            lig_type_l.append(b["lig_atom_type"])
            lig_mask_l.append(torch.zeros(Na, dtype=torch.bool))

        lig_ctr_l.append(b["lig_center"])
        aff_l.append(b["affinity"])
        bind_l.append(b["bind_label"])
        names.append(b["name"])
        ei_list.append(b["lig_edge_index"])
        et_list.append(b["lig_edge_type"])

    return dict(
        rec_feats     = torch.stack(rec_feats_l),     # (B, Tr, K, 6)
        rec_centers   = torch.stack(rec_centers_l),   # (B, Tr, 3)
        pocket_labels = torch.stack(pocket_l),         # (B, Tr)
        rec_mask      = torch.stack(rec_mask_l),       # (B, Tr)  True=pad
        lig_pos       = torch.stack(lig_pos_l),        # (B, Na, 3)
        lig_atom_type = torch.stack(lig_type_l),       # (B, Na)
        lig_mask      = torch.stack(lig_mask_l),       # (B, Na)  True=pad
        lig_edge_index= ei_list,                       # list[Tensor(2,Nb)]
        lig_edge_type = et_list,                       # list[Tensor(Nb,)]
        lig_center    = torch.stack(lig_ctr_l),        # (B, 3)
        affinity      = torch.cat(aff_l),              # (B,)
        bind_label    = torch.cat(bind_l),             # (B,)
        names         = names,
    )
