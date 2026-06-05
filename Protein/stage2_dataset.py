# -*- coding: utf-8 -*-
"""
Stage 2 Dataset: Backbone-aware Surface Dataset for VQ-MAE pretraining.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BackboneAwareSurfaceDataset(Dataset):
    """
    Loads Stage 1 .npz files and yields both surface patches and backbone features.
    """

    def __init__(self, root, seq_len=512, K=50, mask_ratio=0.6, max_residues=256):
        super().__init__()
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npz')]
        if not self.files:
            raise FileNotFoundError(f"No .npz found under {root}")
        self.seq_len = seq_len
        self.K = K
        self.mask_ratio = mask_ratio
        self.max_residues = max_residues
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        if path in self._cache:
            return self._cache[path]
        with np.load(path, allow_pickle=True) as data:
            out = {
                'xs': data['xs'].astype(np.float32),
                'ns': data['ns'].astype(np.float32),
                'patch_centers': data['patch_centers'].astype(np.float32),
                'patch_knn_idx': data['patch_knn_idx'].astype(np.int64),
                'patch_order': data['patch_order'].astype(np.int64),
                'backbone_ncac': data['backbone_ncac'].astype(np.float32),
                'bb_valid_mask': data['bb_valid_mask'].astype(bool),
                'ca_pos': data['ca_pos'].astype(np.float32),
                'ca_type': data['ca_type'].astype(np.int64),
                'torsion_sincos': data['torsion_sincos'].astype(np.float32),
                'torsion_valid_mask': data['torsion_valid_mask'].astype(bool),
                'ss_labels': data['ss_labels'].astype(np.int64),
                'surf2res_idx': data['surf2res_idx'].astype(np.int64),
            }
        knn = out['patch_knn_idx']
        K0 = knn.shape[1]
        if K0 < self.K:
            pad = np.tile(knn[:, -1:], (1, self.K - K0))
            out['patch_knn_idx'] = np.concatenate([knn, pad], axis=1)
        elif K0 > self.K:
            out['patch_knn_idx'] = knn[:, :self.K]
        self._cache[path] = out
        return out

    def __getitem__(self, idx):
        arr = self._load(self.files[idx % len(self.files)])
        xs, ns = arr['xs'], arr['ns']
        centers = arr['patch_centers']
        knn = arr['patch_knn_idx']
        order = arr['patch_order']
        surf2res = arr['surf2res_idx']

        Nc = centers.shape[0]
        if Nc <= self.seq_len:
            sel = order
        else:
            start = np.random.randint(0, Nc - self.seq_len + 1)
            sel = order[start:start + self.seq_len]

        pts_idx = knn[sel]
        ctrs = centers[sel]
        rel_xyz = xs[pts_idx] - ctrs[:, None, :]
        norms = ns[pts_idx]
        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
        coords = rel_xyz.astype(np.float32)

        T = feats.shape[0]
        mask = np.zeros(T, dtype=np.bool_)
        num_mask = max(1, int(round(self.mask_ratio * T)))
        mask[np.random.choice(T, size=num_mask, replace=False)] = True

        s2r = surf2res[sel]

        bb_ncac = arr['backbone_ncac']
        ca_pos = arr['ca_pos']
        ca_type = arr['ca_type']
        torsion_sc = arr['torsion_sincos']
        torsion_valid = arr['torsion_valid_mask']
        ss_labels = arr['ss_labels']
        bb_valid = arr['bb_valid_mask']

        L = ca_pos.shape[0]
        L_use = min(L, self.max_residues)

        bb_feats = np.zeros((L_use, 35), dtype=np.float32)
        bb_mask = np.zeros(L_use, dtype=np.bool_)

        for i in range(L_use):
            if not bb_valid[i]:
                continue
            n_rel = bb_ncac[i, 0] - ca_pos[i]
            c_rel = bb_ncac[i, 2] - ca_pos[i]
            if i < L_use - 1 and bb_valid[i + 1]:
                n_next_rel = bb_ncac[i + 1, 0] - ca_pos[i]
            else:
                n_next_rel = np.zeros(3, dtype=np.float32)

            bb_feats[i, :3] = n_rel
            bb_feats[i, 3:6] = c_rel
            bb_feats[i, 6:9] = n_next_rel
            bb_feats[i, 9:15] = torsion_sc[i]

            aa_idx = int(ca_type[i])
            if 0 <= aa_idx < 20:
                bb_feats[i, 15 + aa_idx] = 1.0
            bb_mask[i] = True

        return {
            'surface_feats': torch.from_numpy(feats),
            'surface_coords': torch.from_numpy(coords),
            'surface_centers': torch.from_numpy(ctrs),
            'surface_mask': torch.from_numpy(mask),
            'bb_feats': torch.from_numpy(bb_feats),
            'bb_torsion_target': torch.from_numpy(torsion_sc[:L_use]),
            'bb_ss_target': torch.from_numpy(ss_labels[:L_use]),
            'bb_valid': torch.from_numpy(bb_mask),
            'torsion_valid': torch.from_numpy(torsion_valid[:L_use]),
            'surf2res': torch.from_numpy(s2r),
        }


def stage2_collate_fn(batch):
    """Pad surface and backbone sequences to batch max."""
    B = len(batch)
    K = batch[0]['surface_feats'].shape[1]
    T_max = max(b['surface_feats'].shape[0] for b in batch)
    L_max = max(b['bb_feats'].shape[0] for b in batch)

    sf = torch.zeros(B, T_max, K, 6)
    sc = torch.zeros(B, T_max, K, 3)
    sctrs = torch.zeros(B, T_max, 3)
    smask = torch.zeros(B, T_max, dtype=torch.bool)
    spad = torch.ones(B, T_max, dtype=torch.bool)

    bbf = torch.zeros(B, L_max, 35)
    bbt = torch.zeros(B, L_max, 6)
    bbss = torch.zeros(B, L_max, dtype=torch.long)
    bbv = torch.zeros(B, L_max, dtype=torch.bool)
    tv = torch.zeros(B, L_max, dtype=torch.bool)
    s2r = torch.zeros(B, T_max, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b['surface_feats'].shape[0]
        L = b['bb_feats'].shape[0]

        sf[i, :T] = b['surface_feats']
        sc[i, :T] = b['surface_coords']
        sctrs[i, :T] = b['surface_centers']
        smask[i, :T] = b['surface_mask']
        spad[i, :T] = False

        bbf[i, :L] = b['bb_feats']
        bbt[i, :L] = b['bb_torsion_target']
        bbss[i, :L] = b['bb_ss_target']
        bbv[i, :L] = b['bb_valid']
        tv[i, :L] = b['torsion_valid']
        s2r[i, :T] = b['surf2res'].clamp(max=L - 1)

    return {
        'surface_feats': sf, 'surface_coords': sc, 'surface_centers': sctrs,
        'surface_mask': smask, 'surface_pad': spad,
        'bb_feats': bbf, 'bb_torsion_target': bbt, 'bb_ss_target': bbss,
        'bb_valid': bbv, 'torsion_valid': tv, 'surf2res': s2r,
    }
