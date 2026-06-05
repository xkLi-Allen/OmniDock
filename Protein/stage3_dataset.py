# -*- coding: utf-8 -*-
"""
Stage 3 Dataset: SKEMPI docking dataset with backbone information.

This version keeps the original Stage 3 behavior by default, but adds two
Stage-4-friendly features:
  1) interface-centered surface sampling via surface_sampling='interface'
  2) ligand/receptor torsion_valid_mask return and collation
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SkempiDockingDataset(Dataset):
    """
    SKEMPI v2 docking dataset with backbone-aware features.
    Loads receptor and ligand surfaces + backbone information.

    Args:
        surface_sampling:
            'random'    - original Morton-window sampling, suitable for Stage 3.
            'interface' - choose surface windows closest to the partner chain,
                          recommended for Stage 4 generator training.
    """

    GAS_CONST = 1.987e-3  # kcal / (mol*K)

    def __init__(self, skempi_csv, npz_root, K=50, seq_len=512, max_residues=256,
                 interface_cutoff=8.0, pocket_margin=2.0, cache_npz=True,
                 surface_sampling="random", pocket_target_type="surface_midpoint_robust",
                 smooth_ss_labels=False, ss_min_segment_len=3,
                 ss_label_source="auto"):
        super().__init__()
        assert os.path.isfile(skempi_csv), f"CSV not found: {skempi_csv}"
        assert os.path.isdir(npz_root), f"npz_root not found: {npz_root}"
        if surface_sampling not in {"random", "interface"}:
            raise ValueError("surface_sampling must be 'random' or 'interface'")
        if pocket_target_type not in {"surface_midpoint_robust", "native_ligand_ca", "native_interface_ca"}:
            raise ValueError("pocket_target_type must be surface_midpoint_robust, native_ligand_ca, or native_interface_ca")

        self.npz_root = npz_root
        self.K = K
        self.seq_len = seq_len
        self.max_residues = max_residues
        self.interface_cutoff = interface_cutoff
        self.pocket_margin = pocket_margin
        self.cache_npz = cache_npz
        self.surface_sampling = surface_sampling
        self.pocket_target_type = pocket_target_type
        self.smooth_ss_labels = bool(smooth_ss_labels)
        self.ss_min_segment_len = int(ss_min_segment_len)
        if ss_label_source not in {"auto", "dssp", "clean", "original", "none"}:
            raise ValueError("ss_label_source must be auto, dssp, clean, original, or none")
        self.ss_label_source = ss_label_source
        self._warned_missing_dssp = False
        self._warned_low_dssp = False
        self._cache = {}

        df = pd.read_csv(skempi_csv, sep=";")
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce").fillna(298.0)
        df = df[(df["Affinity_mut_parsed"] > 0) & (df["Affinity_wt_parsed"] > 0)]
        self.df = df.reset_index(drop=True)

        self.samples = []
        for row_idx, row in self.df.iterrows():
            try:
                pdb_id, rec_chains, lig_chains = self._parse_pdb_field(row["#Pdb"])
            except ValueError:
                continue
            rec_npz = self._find_npz(pdb_id, rec_chains)
            lig_npz = self._find_npz(pdb_id, lig_chains)
            if rec_npz and lig_npz:
                self.samples.append((row_idx, rec_npz, lig_npz, row["#Pdb"]))

        print(f"[SkempiDockingDataset] usable samples = {len(self.samples)}")
        print(f"[SkempiDockingDataset] surface_sampling = {self.surface_sampling}")
        print(f"[SkempiDockingDataset] pocket_target_type = {self.pocket_target_type}")
        print(f"[SkempiDockingDataset] ss_label_source = {self.ss_label_source}")

    @staticmethod
    def _parse_pdb_field(pdb_str):
        parts = str(pdb_str).split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected format: {pdb_str}")
        return parts[0], parts[1], parts[2]

    def _find_npz(self, pdb_id, chains):
        path = os.path.join(self.npz_root, f"{pdb_id}_{chains}.npz")
        return path if os.path.isfile(path) else None

    def _load_npz(self, path):
        if self.cache_npz and path in self._cache:
            return self._cache[path]
        with np.load(path, allow_pickle=True) as data:
            out = {k: data[k] for k in data.files}
        knn = out['patch_knn_idx']
        K0 = knn.shape[1]
        if K0 < self.K:
            out['patch_knn_idx'] = np.concatenate(
                [knn, np.tile(knn[:, -1:], (1, self.K - K0))], axis=1)
        elif K0 > self.K:
            out['patch_knn_idx'] = knn[:, :self.K]
        if self.cache_npz:
            self._cache[path] = out
        return out

    def _surface_from_selection(self, arr, sel):
        xs = arr['xs'].astype(np.float32)
        ns = arr['ns'].astype(np.float32)
        centers = arr['patch_centers'].astype(np.float32)
        knn = arr['patch_knn_idx'].astype(np.int64)

        sel = np.asarray(sel, dtype=np.int64)
        pts_idx = knn[sel]
        ctrs = centers[sel]
        rel_xyz = xs[pts_idx] - ctrs[:, None, :]
        norms = ns[pts_idx]
        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
        return feats, ctrs, sel

    def _sample_surface_window(self, arr):
        centers = arr['patch_centers'].astype(np.float32)
        order = arr['patch_order'].astype(np.int64)

        Nc = centers.shape[0]
        if Nc <= self.seq_len:
            sel = order
        else:
            start = np.random.randint(0, Nc - self.seq_len + 1)
            sel = order[start:start + self.seq_len]
        return self._surface_from_selection(arr, sel)

    def _sample_interface_surface_window(self, query_arr, partner_arr):
        """Sample query-chain patches closest to partner-chain surface patches."""
        centers = query_arr['patch_centers'].astype(np.float32)
        partner_centers = partner_arr['patch_centers'].astype(np.float32)
        order = query_arr['patch_order'].astype(np.int64)

        Nc = centers.shape[0]
        if Nc <= self.seq_len or partner_centers.shape[0] == 0:
            sel = order[:min(Nc, self.seq_len)]
            return self._surface_from_selection(query_arr, sel)

        # Compute min distance in chunks to avoid a large Nc x Npartner allocation.
        min_dist = np.full((Nc,), np.inf, dtype=np.float32)
        chunk = 1024
        for st in range(0, Nc, chunk):
            ed = min(st + chunk, Nc)
            diff = centers[st:ed, None, :] - partner_centers[None, :, :]
            min_dist[st:ed] = np.linalg.norm(diff, axis=-1).min(axis=1).astype(np.float32)

        # Closest patches give the interface. Sorting is deterministic and stable.
        sel = np.argsort(min_dist)[:self.seq_len]
        return self._surface_from_selection(query_arr, sel)

    @staticmethod
    def _smooth_ss_sequence(ss_labels, min_len=3):
        ss = np.asarray(ss_labels, dtype=np.int64).copy()
        L = ss.shape[0]
        start = 0
        while start < L:
            val = int(ss[start])
            end = start + 1
            while end < L and int(ss[end]) == val:
                end += 1
            if val in (0, 1) and end - start < min_len:
                ss[start:end] = 2
            start = end
        return ss

    def _select_ss_labels(self, arr, L_expected):
        source_used = "none"
        if self.ss_label_source == "none":
            return np.full(L_expected, 2, dtype=np.int64), np.zeros(L_expected, dtype=bool), source_used

        def original_labels():
            if 'ss_labels_clean' in arr and self.ss_label_source != "original":
                return arr['ss_labels_clean'].astype(np.int64), np.ones(arr['ss_labels_clean'].shape[0], dtype=bool), "clean"
            if 'ss_labels' in arr:
                return arr['ss_labels'].astype(np.int64), np.ones(arr['ss_labels'].shape[0], dtype=bool), "original"
            return np.full(L_expected, 2, dtype=np.int64), np.zeros(L_expected, dtype=bool), "missing"

        use_dssp = self.ss_label_source in {"auto", "dssp"} and 'ss_labels_dssp' in arr
        if use_dssp:
            labels = arr['ss_labels_dssp'].astype(np.int64)
            valid = arr['ss_valid_mask_dssp'].astype(bool) if 'ss_valid_mask_dssp' in arr else np.ones(labels.shape[0], dtype=bool)
            ratio = float(valid[:L_expected].mean()) if valid.size else 0.0
            if self.ss_label_source == "auto" and ratio < 0.5:
                if not self._warned_low_dssp:
                    print(f"[SkempiDockingDataset][WARNING] DSSP valid ratio too low ({ratio:.3f}); auto fallback to original/clean")
                    self._warned_low_dssp = True
                labels, valid, source_used = original_labels()
            else:
                source_used = "dssp"
        elif self.ss_label_source == "dssp":
            if not self._warned_missing_dssp:
                print("[SkempiDockingDataset][WARNING] ss_label_source=dssp requested but ss_labels_dssp is missing; falling back to clean/original labels")
                self._warned_missing_dssp = True
            labels, valid, source_used = original_labels()
        else:
            labels, valid, source_used = original_labels()

        labels = labels[:L_expected]
        valid = valid[:L_expected]
        if self.smooth_ss_labels:
            labels = self._smooth_ss_sequence(labels, min_len=self.ss_min_segment_len)
        return labels, valid, source_used

    def _build_backbone_feats(self, arr):
        bb_ncac = arr['backbone_ncac'].astype(np.float32)
        ca_pos = arr['ca_pos'].astype(np.float32)
        ca_type = arr['ca_type'].astype(np.int64)
        torsion_sc = arr['torsion_sincos'].astype(np.float32)
        bb_valid = arr['bb_valid_mask'].astype(bool)
        ss_labels, ss_valid, ss_source_used = self._select_ss_labels(arr, ca_pos.shape[0])

        if 'torsion_valid_mask' in arr:
            torsion_valid = arr['torsion_valid_mask'].astype(bool)
        else:
            torsion_valid = bb_valid.copy()
            if torsion_valid.shape[0] > 0:
                torsion_valid[0] = False
                torsion_valid[-1] = False

        L = min(ca_pos.shape[0], self.max_residues)
        bb_feats = np.zeros((L, 35), dtype=np.float32)
        bb_mask = np.zeros(L, dtype=np.bool_)

        for i in range(L):
            if not bb_valid[i]:
                continue
            bb_feats[i, :3] = bb_ncac[i, 0] - ca_pos[i]
            bb_feats[i, 3:6] = bb_ncac[i, 2] - ca_pos[i]
            if i < L - 1 and bb_valid[i + 1]:
                bb_feats[i, 6:9] = bb_ncac[i + 1, 0] - ca_pos[i]
            bb_feats[i, 9:15] = torsion_sc[i]
            aa_idx = int(ca_type[i])
            if 0 <= aa_idx < 20:
                bb_feats[i, 15 + aa_idx] = 1.0
            bb_mask[i] = True

        torsion_valid = torsion_valid[:L] & bb_mask
        ss_valid = ss_valid[:L] & bb_mask
        return bb_feats, bb_mask, torsion_sc[:L], torsion_valid, ss_labels[:L], ss_valid, ca_pos[:L], ss_source_used

    def _compute_pocket(self, rec_centers, lig_centers, lig_ca=None, rec_ca=None):
        if self.pocket_target_type == "native_ligand_ca" and lig_ca is not None and lig_ca.shape[0] > 0:
            center = lig_ca.mean(axis=0).astype(np.float32)
            radial = np.linalg.norm(lig_ca - center, axis=-1)
            radius = float(np.percentile(radial, 90.0) + self.pocket_margin)
            radius = float(np.clip(radius, 6.0, 30.0))
            return center, np.array([radius], dtype=np.float32)

        if self.pocket_target_type == "native_interface_ca" and lig_ca is not None and rec_ca is not None and lig_ca.shape[0] > 0 and rec_ca.shape[0] > 0:
            dist_ca = np.linalg.norm(lig_ca[:, None, :] - rec_ca[None, :, :], axis=-1)
            iface_mask = dist_ca.min(axis=1) < self.interface_cutoff
            if iface_mask.sum() >= 3:
                iface_ca = lig_ca[iface_mask]
            else:
                iface_ca = lig_ca
            center = iface_ca.mean(axis=0).astype(np.float32)
            radial = np.linalg.norm(iface_ca - center, axis=-1)
            radius = float(np.percentile(radial, 90.0) + self.pocket_margin)
            radius = float(np.clip(radius, 6.0, 30.0))
            return center, np.array([radius], dtype=np.float32)

        if rec_centers.shape[0] == 0 or lig_centers.shape[0] == 0:
            return np.zeros(3, dtype=np.float32), np.array([8.0], dtype=np.float32)
        diff = rec_centers[:, None, :] - lig_centers[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        mask = dist < self.interface_cutoff
        if not mask.any():
            idx = np.unravel_index(dist.argmin(), dist.shape)
            rec_pts = rec_centers[idx[0:1]]
            lig_pts = lig_centers[idx[1:2]]
        else:
            rec_idx, lig_idx = np.where(mask)
            rec_pts = rec_centers[rec_idx]
            lig_pts = lig_centers[lig_idx]
        mid = 0.5 * (rec_pts + lig_pts)
        center = mid.mean(axis=0).astype(np.float32)
        radial = np.linalg.norm(mid - center, axis=-1)
        # Protein-protein interfaces can be elongated; using the max distance
        # turns pocket radius into a whole-interface diameter and dominates
        # Stage 4 overfit. A robust radius keeps the target local enough for
        # ligand-like backbone placement while still covering most contacts.
        radius = float(np.percentile(radial, 75.0) + self.pocket_margin)
        radius = float(np.clip(radius, 6.0, 30.0))
        return center, np.array([radius], dtype=np.float32)

    def _compute_ddG(self, kd_mut, kd_wt, T):
        if kd_mut <= 0 or kd_wt <= 0:
            return 0.0
        return self.GAS_CONST * T * (math.log(kd_mut) - math.log(kd_wt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, rec_npz, lig_npz, pdb_str = self.samples[idx]
        row = self.df.iloc[row_idx]

        rec = self._load_npz(rec_npz)
        lig = self._load_npz(lig_npz)

        if self.surface_sampling == "interface":
            rec_feats, rec_centers, _ = self._sample_interface_surface_window(rec, lig)
            lig_feats, lig_centers, _ = self._sample_interface_surface_window(lig, rec)
        else:
            rec_feats, rec_centers, _ = self._sample_surface_window(rec)
            lig_feats, lig_centers, _ = self._sample_surface_window(lig)

        rec_bb_feats, rec_bb_mask, _, rec_torsion_valid, _, rec_ss_valid, rec_ca, rec_ss_source = self._build_backbone_feats(rec)
        lig_bb_feats, lig_bb_mask, lig_torsion, lig_torsion_valid, lig_ss, lig_ss_valid, lig_ca, lig_ss_source = self._build_backbone_feats(lig)

        pocket_center, pocket_radius = self._compute_pocket(
            rec['patch_centers'].astype(np.float32),
            lig['patch_centers'].astype(np.float32),
            lig_ca=lig_ca.astype(np.float32),
            rec_ca=rec_ca.astype(np.float32),
        )

        ddG = self._compute_ddG(row["Affinity_mut_parsed"], row["Affinity_wt_parsed"], row["Temperature"])

        return {
            'rec_feats': torch.from_numpy(rec_feats),
            'rec_centers': torch.from_numpy(rec_centers),
            'lig_feats': torch.from_numpy(lig_feats),
            'lig_centers': torch.from_numpy(lig_centers),
            'rec_bb_feats': torch.from_numpy(rec_bb_feats),
            'rec_bb_mask': torch.from_numpy(rec_bb_mask),
            'rec_torsion_valid': torch.from_numpy(rec_torsion_valid),
            'lig_bb_feats': torch.from_numpy(lig_bb_feats),
            'lig_bb_mask': torch.from_numpy(lig_bb_mask),
            'lig_torsion_valid': torch.from_numpy(lig_torsion_valid),
            'lig_torsion_target': torch.from_numpy(lig_torsion),
            'lig_ss_target': torch.from_numpy(lig_ss),
            'lig_ss_valid': torch.from_numpy(lig_ss_valid),
            'ss_source_used': lig_ss_source,
            'lig_ca_pos': torch.from_numpy(lig_ca),
            'pocket_center': torch.from_numpy(pocket_center),
            'pocket_radius': torch.from_numpy(pocket_radius),
            'bind_label': torch.tensor([1.0], dtype=torch.float32),
            'affinity': torch.tensor([ddG], dtype=torch.float32),
            'name': pdb_str,
        }


def stage3_collate_fn(batch):
    B = len(batch)
    Tr_max = max(b['rec_feats'].shape[0] for b in batch)
    Tl_max = max(b['lig_feats'].shape[0] for b in batch)
    Lr_max = max(b['rec_bb_feats'].shape[0] for b in batch)
    Ll_max = max(b['lig_bb_feats'].shape[0] for b in batch)

    def pad_tensor(tensors, max_len, pad_val=0):
        out = []
        for t in tensors:
            L = t.shape[0]
            if L < max_len:
                pad_shape = (max_len - L,) + t.shape[1:]
                pad = torch.full(pad_shape, pad_val, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    def make_pad_mask(lengths, max_len):
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, L:] = True
        return mask

    rec_feats = pad_tensor([b['rec_feats'] for b in batch], Tr_max)
    rec_centers = pad_tensor([b['rec_centers'] for b in batch], Tr_max)
    rec_pad = make_pad_mask([b['rec_feats'].shape[0] for b in batch], Tr_max)

    lig_feats = pad_tensor([b['lig_feats'] for b in batch], Tl_max)
    lig_centers = pad_tensor([b['lig_centers'] for b in batch], Tl_max)
    lig_pad = make_pad_mask([b['lig_feats'].shape[0] for b in batch], Tl_max)

    rec_bb_feats = pad_tensor([b['rec_bb_feats'] for b in batch], Lr_max)
    rec_bb_mask = pad_tensor([b['rec_bb_mask'] for b in batch], Lr_max)
    rec_torsion_valid = pad_tensor([b['rec_torsion_valid'] for b in batch], Lr_max)

    lig_bb_feats = pad_tensor([b['lig_bb_feats'] for b in batch], Ll_max)
    lig_bb_mask = pad_tensor([b['lig_bb_mask'] for b in batch], Ll_max)
    lig_torsion_valid = pad_tensor([b['lig_torsion_valid'] for b in batch], Ll_max)
    lig_torsion = pad_tensor([b['lig_torsion_target'] for b in batch], Ll_max)
    lig_ss = pad_tensor([b['lig_ss_target'] for b in batch], Ll_max)
    lig_ss_valid = pad_tensor([b.get('lig_ss_valid', torch.ones_like(b['lig_ss_target'], dtype=torch.bool)) for b in batch], Ll_max)
    lig_ca = pad_tensor([b['lig_ca_pos'] for b in batch], Ll_max)

    pocket_center = torch.stack([b['pocket_center'] for b in batch], dim=0)
    pocket_radius = torch.stack([b['pocket_radius'] for b in batch], dim=0)
    bind_label = torch.stack([b['bind_label'] for b in batch], dim=0).squeeze(-1)
    affinity = torch.stack([b['affinity'] for b in batch], dim=0).squeeze(-1)

    return {
        'rec_feats': rec_feats, 'rec_centers': rec_centers, 'rec_pad': rec_pad,
        'lig_feats': lig_feats, 'lig_centers': lig_centers, 'lig_pad': lig_pad,
        'rec_bb_feats': rec_bb_feats, 'rec_bb_mask': rec_bb_mask,
        'rec_torsion_valid': rec_torsion_valid,
        'lig_bb_feats': lig_bb_feats, 'lig_bb_mask': lig_bb_mask,
        'lig_torsion_valid': lig_torsion_valid,
        'lig_torsion_target': lig_torsion, 'lig_ss_target': lig_ss,
        'lig_ss_valid': lig_ss_valid, 'ss_source_used': [b.get('ss_source_used', '') for b in batch], 'lig_ca_pos': lig_ca,
        'pocket_center': pocket_center, 'pocket_radius': pocket_radius,
        'bind_label': bind_label, 'affinity': affinity,
    }
