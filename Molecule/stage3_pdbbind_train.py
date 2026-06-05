# -*- coding: utf-8 -*-
"""
Stage-3: Pocket-conditioned Docking Foundation Model
Support:
  - PDBbind-CN xlsx as reference table (auto header-row detection)
  - Negative-sampled pose CSV (row_idx/pdb_id/rec_chains/lig_chains/pose_type/R_flat/shift[/dG_bind])
  - Per-chain / chain-group surfaces from Stage1: {PDBID}_{CHAIN_GROUP}.npz

Default paths are set to your current project layout.

Author: patched for PDBbind-xlsx + neg-sample csv
"""

import os
import math
import argparse
import re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# 0) Utils
# ============================================================

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean_chain_group(s: str) -> str:
    """
    normalize chain group like:
      "H,L" "H+L" "H L" "H_L" -> "HL"
      remove separators, uppercase, unique+sorted
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    for ch in [",", "+", " ", "_", ":", "|", "/", "\\", ";"]:
        s = s.replace(ch, "")
    s = s.upper()
    # keep alnum only
    s = "".join([c for c in s if c.isalnum()])
    # unique + sorted to match Stage1 output
    s = "".join(sorted(set(list(s))))
    return s


def _is_finite_number(x) -> bool:
    try:
        v = float(x)
        return np.isfinite(v)
    except Exception:
        return False


def random_rotation_matrices(batch_size: int, device: torch.device) -> torch.Tensor:
    """Uniform random SO(3) by axis-angle + Rodrigues."""
    axis = torch.randn(batch_size, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    theta = 2 * math.pi * torch.rand(batch_size, 1, device=device)
    ct = torch.cos(theta).squeeze(1)
    st = torch.sin(theta).squeeze(1)
    vt = 1.0 - ct

    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]

    R = torch.zeros(batch_size, 3, 3, device=device)
    R[:, 0, 0] = ct + kx * kx * vt
    R[:, 0, 1] = kx * ky * vt - kz * st
    R[:, 0, 2] = kx * kz * vt + ky * st

    R[:, 1, 0] = ky * kx * vt + kz * st
    R[:, 1, 1] = ct + ky * ky * vt
    R[:, 1, 2] = ky * kz * vt - kx * st

    R[:, 2, 0] = kz * kx * vt - ky * st
    R[:, 2, 1] = kz * ky * vt + kx * st
    R[:, 2, 2] = ct + kz * kz * vt
    return R


# ============================================================
# 1) PDBbind xlsx loader (AUTO header-row detection)
# ============================================================

_PDB_CODE_COL_PATTERNS = [
    r"\bpdb\s*code\b",
    r"\bpdb\s*id\b",
    r"\bpdb\b",
    r"pdb编号",
    r"pdb\s*码",
]

_AFF_COL_PATTERNS = [
    r"delta\s*g",
    r"Δg",
    r"\bdg\b",
    r"binding\s*free",
    r"binding\s*energy",
    r"\bkd\b",
    r"\bki\b",
    r"ic50",
    r"affinity",
    r"亲和",
    r"解离",
    r"结合自由能",
]


def _score_header_row(row_vals: List[str]) -> float:
    """
    Score a candidate header row: more "pdb/code/affinity/chain" tokens + enough non-empty cells.
    """
    nonempty = [v for v in row_vals if v and v.lower() != "nan"]
    if len(nonempty) < 3:
        return -1e9
    joined = " | ".join(nonempty).lower()

    score = 0.0
    # strong signals
    if "pdb" in joined:
        score += 3.0
    if "code" in joined or "id" in joined or "编号" in joined:
        score += 2.0
    if "chain" in joined or "链" in joined:
        score += 1.0

    # affinity-ish signals
    for pat in _AFF_COL_PATTERNS:
        if re.search(pat, joined, flags=re.IGNORECASE):
            score += 0.7
            break

    # prefer wider rows
    score += min(len(nonempty), 30) / 10.0
    return score


def _make_unique_columns(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for i, c in enumerate(cols):
        c = str(c).strip()
        if not c or c.lower() == "nan":
            c = f"col_{i}"
        base = c
        if base in seen:
            seen[base] += 1
            c = f"{base}__{seen[base]}"
        else:
            seen[base] = 0
        out.append(c)
    return out


def _find_pdb_col(df: pd.DataFrame) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    for c in cols:
        c0 = str(c).strip()
        for pat in _PDB_CODE_COL_PATTERNS:
            if re.search(pat, c0, flags=re.IGNORECASE):
                return c
    # fallback: detect by values looking like 4-char PDB ids
    for c in cols:
        s = df[c].astype(str).str.strip()
        # count how many match typical pdb id
        hit = s.str.match(r"^[0-9][A-Za-z0-9]{3}$").mean()
        if hit > 0.5:
            return c
    return None


def load_pdbbind_xlsx(
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    max_scan_rows: int = 80,
) -> pd.DataFrame:
    """
    Robustly load PDBbind-CN xlsx where the first row may be a title ("Search results ..."),
    and actual table header starts later.

    It scans top rows to detect the best header row.
    """
    assert os.path.isfile(xlsx_path), f"xlsx not found: {xlsx_path}"

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = xls.sheet_names

    cand_sheets = [sheet_name] if sheet_name else sheets

    last_err = None
    for sh in cand_sheets:
        try:
            raw = pd.read_excel(xlsx_path, sheet_name=sh, header=None, engine="openpyxl")
            if raw.shape[0] == 0:
                continue

            # find best header row
            best_i = None
            best_score = -1e18
            scan_n = min(max_scan_rows, raw.shape[0])
            for i in range(scan_n):
                row_vals = [str(x).strip() for x in raw.iloc[i].tolist()]
                sc = _score_header_row(row_vals)
                if sc > best_score:
                    best_score = sc
                    best_i = i

            if best_i is None:
                continue

            header = [str(x).strip() for x in raw.iloc[best_i].tolist()]
            header = _make_unique_columns(header)

            df = raw.iloc[best_i + 1:].copy()
            df.columns = header
            df = df.dropna(axis=0, how="all").reset_index(drop=True)

            # locate pdb col
            pdb_col = _find_pdb_col(df)
            if pdb_col is None:
                # if this sheet doesn't contain pdb table, try next
                continue

            return df

        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise ValueError(f"Failed to load any usable sheet from xlsx: {xlsx_path}. last_err={last_err}")
    raise ValueError(
        f"Cannot detect usable table from xlsx: {xlsx_path}. "
        f"Tried sheets={cand_sheets}, but no sheet contained a detectable PDB column."
    )


# ============================================================
# 2) Dataset: Negative-sampled pose CSV for PDBbind
# ============================================================

class PDBbindNegSampleCSVDataset(Dataset):
    """
    Train CSV (comma-separated) example columns:
      row_idx,pdb_str,pdb_id,rec_chains,lig_chains,pose_type,neg_id,R_flat,shift,dG_bind

    - rec/lig surfaces should exist in npz_root as:
        {PDBID}_{RECCHAINS}.npz and {PDBID}_{LIGCHAINS}.npz
      where CHAINS are normalized by _clean_chain_group.

    Affinity target:
      - default: use pose csv's dG_bind for pos samples only (neg -> 0)
      - optional: use ref xlsx (by row_idx) if --aff_source=ref_xlsx
    """

    def __init__(
        self,
        train_csv: str,
        npz_root: str,
        K: int = 50,
        seq_len: int = 512,
        cache_npz: bool = True,
        interface_cutoff: float = 8.0,
        pocket_margin: float = 2.0,
        # ref xlsx (optional)
        ref_xlsx: Optional[str] = None,
        ref_sheet: Optional[str] = None,
        aff_source: str = "pose_csv",  # pose_csv | ref_xlsx
        ref_aff_col: Optional[str] = None,
        aff_clip: float = 200.0,
    ):
        super().__init__()
        assert os.path.isfile(train_csv), f"train_csv not found: {train_csv}"
        assert os.path.isdir(npz_root), f"npz_root not found: {npz_root}"
        if aff_source == "ref_xlsx":
            assert ref_xlsx is not None and os.path.isfile(ref_xlsx), f"ref_xlsx not found: {ref_xlsx}"

        self.train_csv = train_csv
        self.npz_root = npz_root
        self.K = K
        self.seq_len = seq_len
        self.cache_npz = cache_npz
        self.interface_cutoff = interface_cutoff
        self.pocket_margin = pocket_margin
        self.aff_source = aff_source
        self.ref_xlsx = ref_xlsx
        self.ref_sheet = ref_sheet
        self.ref_aff_col = ref_aff_col
        self.aff_clip = float(aff_clip)

        df_pose = pd.read_csv(train_csv)  # comma-separated
        req = {"row_idx", "pose_type", "R_flat", "shift"}
        miss = req - set(df_pose.columns)
        if miss:
            raise ValueError(f"train_csv missing columns: {sorted(list(miss))}")
        # prefer explicit fields
        if not {"pdb_id", "rec_chains", "lig_chains"}.issubset(set(df_pose.columns)):
            raise ValueError("train_csv must contain pdb_id, rec_chains, lig_chains for PDBbind mode.")

        self.df_pose = df_pose.reset_index(drop=True)

        # load ref (optional)
        self.aff_by_rowidx = None
        self._ref_mode_ok = False
        if self.aff_source == "ref_xlsx":
            df_ref = load_pdbbind_xlsx(ref_xlsx, sheet_name=ref_sheet).reset_index(drop=True)

            pdb_col = _find_pdb_col(df_ref)
            if pdb_col is None:
                raise ValueError(f"Cannot find PDB column after loading xlsx. cols={df_ref.columns.tolist()}")

            # choose affinity column
            if ref_aff_col is not None and ref_aff_col in df_ref.columns:
                aff_col = ref_aff_col
            else:
                aff_col = None
                # heuristic find by name
                for c in df_ref.columns:
                    c0 = str(c).strip().lower()
                    if any(re.search(p, c0, flags=re.IGNORECASE) for p in _AFF_COL_PATTERNS):
                        aff_col = c
                        break
                if aff_col is None:
                    # fallback: try numeric-ish columns
                    num_cols = []
                    for c in df_ref.columns:
                        if c == pdb_col:
                            continue
                        x = pd.to_numeric(df_ref[c], errors="coerce")
                        if x.notna().mean() > 0.5:
                            num_cols.append((c, x.notna().mean()))
                    if num_cols:
                        num_cols.sort(key=lambda t: -t[1])
                        aff_col = num_cols[0][0]

            if aff_col is None:
                raise ValueError(
                    f"Cannot auto-detect affinity column in xlsx. "
                    f"Please pass --ref_aff_col. cols={df_ref.columns.tolist()}"
                )

            x = pd.to_numeric(df_ref[aff_col], errors="coerce").fillna(0.0).astype(np.float32).values
            # clip to prevent AMP overflow
            x = np.clip(x, -self.aff_clip, self.aff_clip)
            self.aff_by_rowidx = x
            self._ref_mode_ok = True

            # small log
            usable_pdb = df_ref[pdb_col].astype(str).str.strip().str.match(r"^[0-9][A-Za-z0-9]{3}$").sum()
            print(f"[Dataset] Detected PDBbind-xlsx ref. usable pdb_ids={int(usable_pdb)} | aff_col={aff_col}")

        self._npz_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # build samples
        self.samples: List[Tuple[int, int, str, str, str, str, str]] = []
        self.bind_labels: List[int] = []
        total = len(self.df_pose)
        skipped = 0

        for pose_i, row in self.df_pose.iterrows():
            row_idx_ref = int(row["row_idx"])
            pose_type = str(row["pose_type"]).lower().strip()
            pdb_id = str(row["pdb_id"]).strip()[:4].upper()
            rec_chains = _clean_chain_group(str(row["rec_chains"]))
            lig_chains = _clean_chain_group(str(row["lig_chains"]))

            if not pdb_id or not rec_chains or not lig_chains:
                skipped += 1
                continue

            rec_npz = self._find_npz(pdb_id, rec_chains)
            lig_npz = self._find_npz(pdb_id, lig_chains)
            if rec_npz is None or lig_npz is None:
                skipped += 1
                continue

            R_flat = str(row["R_flat"])
            shift = str(row["shift"])
            self.samples.append((pose_i, row_idx_ref, rec_npz, lig_npz, pose_type, R_flat, shift))
            self.bind_labels.append(1 if pose_type == "pos" else 0)

        print(f"[Dataset] pose_rows={total}, usable={len(self.samples)}, skipped_missing_npz={skipped}")
        if len(self.samples) == 0:
            raise RuntimeError("No usable samples found. Check npz_root naming or Stage1 outputs.")

    def _find_npz(self, pdb_id: str, chains: str) -> Optional[str]:
        pidU = pdb_id.upper()
        g = _clean_chain_group(chains)
        cand = [
            os.path.join(self.npz_root, f"{pidU}_{g}.npz"),
            os.path.join(self.npz_root, f"{pidU}_{g.lower()}.npz"),
            os.path.join(self.npz_root, f"{pidU.lower()}_{g}.npz"),
            os.path.join(self.npz_root, f"{pidU.lower()}_{g.lower()}.npz"),
        ]
        for p in cand:
            if os.path.isfile(p):
                return p
        return None

    def _load_surface(self, path: str) -> Dict[str, np.ndarray]:
        if self.cache_npz and path in self._npz_cache:
            return self._npz_cache[path]

        with np.load(path, allow_pickle=True) as data:
            xs = data["xs"].astype(np.float32)
            ns = data["ns"].astype(np.float32)
            centers = data["patch_centers"].astype(np.float32)
            knn = data["patch_knn_idx"].astype(np.int64)
            if "patch_order" in data:
                order = data["patch_order"].astype(np.int64)
            elif "patch_morton" in data:
                order = data["patch_morton"].astype(np.int64)
            else:
                raise KeyError(f"{path} has no patch_order/patch_morton")

        K0 = knn.shape[1]
        if K0 < self.K:
            pad = np.tile(knn[:, -1:], (1, self.K - K0))
            knn = np.concatenate([knn, pad], axis=1)
        elif K0 > self.K:
            knn = knn[:, :self.K]

        out = dict(xs=xs, ns=ns, centers=centers, knn=knn, order=order)
        if self.cache_npz:
            self._npz_cache[path] = out
        return out

    def _sample_window(self, xs, ns, centers, knn, order) -> Tuple[np.ndarray, np.ndarray]:
        Nc = centers.shape[0]
        if Nc <= self.seq_len:
            sel = order
        else:
            start = np.random.randint(0, Nc - self.seq_len + 1)
            sel = order[start:start + self.seq_len]
        pts_idx = knn[sel]  # (T,K)
        ctrs = centers[sel]  # (T,3)
        rel_xyz = xs[pts_idx] - ctrs[:, None, :]
        norms = ns[pts_idx]
        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)  # (T,K,6)
        return feats, ctrs.astype(np.float32)

    def _compute_pocket_from_interface(self, rec_centers_all: np.ndarray, lig_centers_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if rec_centers_all.shape[0] == 0 or lig_centers_all.shape[0] == 0:
            return np.zeros(3, dtype=np.float32), np.array([8.0], dtype=np.float32)

        diff = rec_centers_all[:, None, :] - lig_centers_all[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        mask = dist < self.interface_cutoff
        if not mask.any():
            idx = np.unravel_index(dist.argmin(), dist.shape)
            rec_pts = rec_centers_all[idx[0:1]]
            lig_pts = lig_centers_all[idx[1:2]]
        else:
            rec_idx, lig_idx = np.where(mask)
            rec_pts = rec_centers_all[rec_idx]
            lig_pts = lig_centers_all[lig_idx]

        mid = 0.5 * (rec_pts + lig_pts)
        center = mid.mean(axis=0).astype(np.float32)
        if mid.shape[0] == 1:
            radius = float(self.interface_cutoff + self.pocket_margin)
        else:
            r = np.linalg.norm(mid - center[None, :], axis=-1)
            radius = float(r.max() + self.pocket_margin)
        return center, np.array([radius], dtype=np.float32)

    @staticmethod
    def _parse_float_list(s: str, n: int) -> np.ndarray:
        arr = np.fromstring(str(s), sep=",", dtype=np.float32)
        if arr.size != n:
            raise ValueError(f"Expected {n} floats, got {arr.size}: {str(s)[:120]}")
        return arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pose_i, row_idx_ref, rec_npz, lig_npz, pose_type, R_flat, shift_str = self.samples[idx]
        row_pose = self.df_pose.iloc[pose_i]

        rec = self._load_surface(rec_npz)
        lig = self._load_surface(lig_npz)

        rec_feats, rec_centers = self._sample_window(**rec)
        lig_feats, lig_centers = self._sample_window(**lig)

        pocket_center, pocket_radius = self._compute_pocket_from_interface(rec["centers"], lig["centers"])

        is_pos = 1.0 if pose_type == "pos" else 0.0
        bind_label = np.array([is_pos], dtype=np.float32)

        # affinity target
        aff = 0.0
        if is_pos > 0.5:
            if self.aff_source == "pose_csv" and "dG_bind" in row_pose.index:
                v = row_pose["dG_bind"]
                aff = float(v) if _is_finite_number(v) else 0.0
            elif self.aff_source == "ref_xlsx" and self._ref_mode_ok and self.aff_by_rowidx is not None:
                if 0 <= row_idx_ref < len(self.aff_by_rowidx):
                    aff = float(self.aff_by_rowidx[row_idx_ref])
                else:
                    aff = 0.0
        # clip
        aff = float(np.clip(aff, -self.aff_clip, self.aff_clip))
        affinity = np.array([aff], dtype=np.float32) if is_pos > 0.5 else np.array([0.0], dtype=np.float32)

        # apply stored R,t to ligand for neg samples
        if is_pos < 0.5:
            R = self._parse_float_list(R_flat, 9).reshape(3, 3)   # row-major
            shift = self._parse_float_list(shift_str, 3).reshape(3)

            com = lig_centers.mean(axis=0, keepdims=True).astype(np.float32)
            centers_rel = lig_centers - com
            lig_centers = centers_rel @ R.T + com + shift[None, :]

            rel = lig_feats[..., :3]
            nrm = lig_feats[..., 3:]
            rel = rel @ R.T
            nrm = nrm @ R.T
            lig_feats = np.concatenate([rel, nrm], axis=-1).astype(np.float32)

        neg_id = int(row_pose["neg_id"]) if "neg_id" in row_pose.index else 0
        name = f"{row_pose.get('pdb_str', '')}|{pose_type}|{neg_id}"

        return {
            "rec_feats": torch.from_numpy(rec_feats),
            "rec_centers": torch.from_numpy(rec_centers),
            "lig_feats": torch.from_numpy(lig_feats),
            "lig_centers": torch.from_numpy(lig_centers),
            "pocket_center": torch.from_numpy(pocket_center),
            "pocket_radius": torch.from_numpy(pocket_radius),
            "bind_label": torch.from_numpy(bind_label),
            "affinity": torch.from_numpy(affinity),
            "name": name,
        }


def docking_collate_fn(batch: List[Dict[str, Any]]):
    """pad Tr/Tl to max within batch."""
    B = len(batch)
    K = batch[0]["rec_feats"].shape[1]
    Tr_max = max(b["rec_feats"].shape[0] for b in batch)
    Tl_max = max(b["lig_feats"].shape[0] for b in batch)

    rec_feats_list, rec_centers_list, rec_mask_list = [], [], []
    lig_feats_list, lig_centers_list, lig_mask_list = [], [], []
    pocket_centers, pocket_radii, bind_labels, affinities = [], [], [], []

    for b in batch:
        # rec
        rf = b["rec_feats"]
        rc = b["rec_centers"]
        Tr = rf.shape[0]
        if Tr < Tr_max:
            pad_r = Tr_max - Tr
            rf = torch.cat([rf, torch.zeros((pad_r, K, 6), dtype=rf.dtype)], dim=0)
            rc = torch.cat([rc, torch.zeros((pad_r, 3), dtype=rc.dtype)], dim=0)
            rm = torch.cat([torch.zeros(Tr, dtype=torch.bool), torch.ones(pad_r, dtype=torch.bool)], dim=0)
        else:
            rm = torch.zeros(Tr_max, dtype=torch.bool)
        rec_feats_list.append(rf)
        rec_centers_list.append(rc)
        rec_mask_list.append(rm)

        # lig
        lf = b["lig_feats"]
        lc = b["lig_centers"]
        Tl = lf.shape[0]
        if Tl < Tl_max:
            pad_l = Tl_max - Tl
            lf = torch.cat([lf, torch.zeros((pad_l, K, 6), dtype=lf.dtype)], dim=0)
            lc = torch.cat([lc, torch.zeros((pad_l, 3), dtype=lc.dtype)], dim=0)
            lm = torch.cat([torch.zeros(Tl, dtype=torch.bool), torch.ones(pad_l, dtype=torch.bool)], dim=0)
        else:
            lm = torch.zeros(Tl_max, dtype=torch.bool)
        lig_feats_list.append(lf)
        lig_centers_list.append(lc)
        lig_mask_list.append(lm)

        pocket_centers.append(b["pocket_center"])
        pocket_radii.append(b["pocket_radius"])
        bind_labels.append(b["bind_label"])
        affinities.append(b["affinity"])

    rec_feats = torch.stack(rec_feats_list, dim=0)
    rec_centers = torch.stack(rec_centers_list, dim=0)
    rec_mask = torch.stack(rec_mask_list, dim=0)

    lig_feats = torch.stack(lig_feats_list, dim=0)
    lig_centers = torch.stack(lig_centers_list, dim=0)
    lig_mask = torch.stack(lig_mask_list, dim=0)

    pocket_center = torch.stack(pocket_centers, dim=0)
    pocket_radius = torch.stack(pocket_radii, dim=0)
    bind_label = torch.stack(bind_labels, dim=0).view(B)
    affinity = torch.stack(affinities, dim=0).view(B)

    return (rec_feats, rec_centers, rec_mask,
            lig_feats, lig_centers, lig_mask,
            pocket_center, pocket_radius, bind_label, affinity)


# ============================================================
# 3) Model (reuse your Stage2 modules)
# ============================================================

from unsupervised_pre_training import (  # type: ignore
    PointMLP,
    SurfFormerBlock,
    SurfVQMAE,
)

class SurfaceEncoder(nn.Module):
    def __init__(self, in_dim=6, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.local = PointMLP(in_dim=in_dim, hidden=d_model, out_dim=d_model)
        self.blocks = nn.ModuleList([
            SurfFormerBlock(d_model=d_model, nhead=nhead, dim_ff=4 * d_model, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, centers):
        x = self.local(feats)  # (B,T,D)
        key_padding = None
        for blk in self.blocks:
            x = blk(x, centers, key_padding)
        x = self.norm(x)
        return x


class PocketHead(nn.Module):
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, rec_tokens, rec_mask):
        valid = ~rec_mask
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (rec_tokens * valid.unsqueeze(-1)).sum(dim=1) / denom
        out = self.mlp(pooled)
        center = out[:, :3]
        log_r = out[:, 3:4]
        radius = F.softplus(log_r) + 1e-6
        return center, radius


class PairEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.rec_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers, K=K, dropout=dropout)
        self.lig_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers, K=K, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

    def forward(self, rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask):
        rec_tokens = self.rec_encoder(rec_feats, rec_centers)
        lig_tokens = self.lig_encoder(lig_feats, lig_centers)

        cross, attn = self.cross_attn(
            query=rec_tokens,
            key=lig_tokens,
            value=lig_tokens,
            key_padding_mask=lig_mask,
        )
        cross = self.cross_norm(cross)
        pair_repr = cross.mean(dim=1)
        return {
            "rec_tokens": rec_tokens,
            "lig_tokens": lig_tokens,
            "cross_tokens": cross,
            "pair_repr": pair_repr,
            "cross_attn": attn,
        }


class MultiTaskHead(nn.Module):
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        in_dim = d_model + 4
        self.bind_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.aff_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_repr, pocket_center, pocket_radius):
        pocket_feat = torch.cat([pocket_center, pocket_radius], dim=-1)
        h = torch.cat([pair_repr, pocket_feat], dim=-1)
        bind_logit = self.bind_mlp(h).squeeze(-1)
        affinity = self.aff_mlp(h).squeeze(-1)
        return bind_logit, affinity


class DockingModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.encoder = PairEncoder(d_model=d_model, nhead=nhead, nlayers=nlayers, K=K, dropout=dropout)
        self.pocket_head = PocketHead(d_model=d_model, hidden=d_model)
        self.head = MultiTaskHead(d_model=d_model, hidden=d_model)

    def forward(self, rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask):
        enc_out = self.encoder(rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask)
        rec_tokens = enc_out["rec_tokens"]
        pair_repr = enc_out["pair_repr"]

        pocket_center_pred, pocket_radius_pred = self.pocket_head(rec_tokens, rec_mask)
        bind_logit, affinity_pred = self.head(pair_repr, pocket_center_pred, pocket_radius_pred)

        return {
            "pocket_center_pred": pocket_center_pred,
            "pocket_radius_pred": pocket_radius_pred,
            "bind_logit": bind_logit,
            "affinity_pred": affinity_pred,
            "rec_tokens": rec_tokens,
            "lig_tokens": enc_out["lig_tokens"],
            "rec_centers": rec_centers,
            "lig_centers": lig_centers,
            "rec_mask": rec_mask,
            "lig_mask": lig_mask,
        }


# ============================================================
# 4) Geometry losses
# ============================================================

def surface_complementarity_loss_pocket(
    rec_tokens, lig_tokens, rec_centers, lig_centers,
    rec_mask, lig_mask, pocket_center_pred, pocket_radius_pred,
    contact_thresh=5.0, pocket_extra=2.0,
):
    device = rec_tokens.device
    B, Tr, D = rec_tokens.shape
    total_loss = rec_tokens.new_tensor(0.0)
    used = 0

    center_det = pocket_center_pred.detach()
    radius_det = pocket_radius_pred.detach()

    for b in range(B):
        valid_r = ~rec_mask[b]
        valid_l = ~lig_mask[b]

        r_ctrs = rec_centers[b][valid_r]
        l_ctrs = lig_centers[b][valid_l]
        if r_ctrs.numel() == 0 or l_ctrs.numel() == 0:
            continue

        pc = center_det[b]
        pr = radius_det[b, 0]
        dist_p = torch.norm(r_ctrs - pc.unsqueeze(0), dim=-1)
        in_pocket = dist_p <= (pr + pocket_extra)
        if in_pocket.sum().item() == 0:
            in_pocket = torch.ones_like(dist_p, dtype=torch.bool, device=device)

        r_tok = rec_tokens[b][valid_r][in_pocket]
        r_ctrs_p = r_ctrs[in_pocket]
        if r_tok.numel() == 0:
            continue

        l_tok = lig_tokens[b][valid_l]

        with torch.cuda.amp.autocast(enabled=False):
            Dmat = torch.cdist(r_ctrs_p.float(), l_ctrs.float())
            contact = (Dmat <= float(contact_thresh)).float()

            scores = (r_tok.float() @ l_tok.float().t()) / math.sqrt(D)

            pos_frac = contact.mean().clamp(min=1e-4, max=1.0)
            pos_weight = (1.0 - pos_frac) / pos_frac
            bce = F.binary_cross_entropy_with_logits(scores, contact, pos_weight=pos_weight)

        total_loss = total_loss + bce
        used += 1

    if used == 0:
        return rec_tokens.new_tensor(0.0)
    return total_loss / used


def local_flexibility_loss(tokens, centers, mask, k_neighbors=8):
    B, T, D = tokens.shape
    total = tokens.new_tensor(0.0)
    used = 0
    for b in range(B):
        valid = ~mask[b]
        h = tokens[b][valid]
        c = centers[b][valid]
        Tv = h.shape[0]
        if Tv <= 1:
            continue

        with torch.cuda.amp.autocast(enabled=False):
            Dmat = torch.cdist(c.float(), c.float())
            k = min(k_neighbors + 1, Tv)
            _, nn_idx = torch.topk(Dmat, k=k, dim=-1, largest=False)
            nn_idx = nn_idx[:, 1:]
            diff = h.float().unsqueeze(1) - h.float()[nn_idx]
            l_b = diff.pow(2).mean()

        total = total + l_b
        used += 1
    if used == 0:
        return tokens.new_tensor(0.0)
    return total / used


# ============================================================
# 5) Loss aggregation (float32-safe)
# ============================================================

def compute_losses(model_out,
                   pocket_center_gt, pocket_radius_gt,
                   bind_label, affinity_gt,
                   is_pos: torch.Tensor,
                   args):
    pc_pred = model_out["pocket_center_pred"]
    pr_pred = model_out["pocket_radius_pred"]
    bind_logit = model_out["bind_logit"]
    aff_pred = model_out["affinity_pred"]

    rec_tokens = model_out["rec_tokens"]
    lig_tokens = model_out["lig_tokens"]
    rec_centers = model_out["rec_centers"]
    lig_centers = model_out["lig_centers"]
    rec_mask = model_out["rec_mask"]
    lig_mask = model_out["lig_mask"]

    # pocket regression
    with torch.cuda.amp.autocast(enabled=False):
        l_pocket_center = F.mse_loss(pc_pred.float(), pocket_center_gt.float())
        l_pocket_radius = F.mse_loss(pr_pred.float(), pocket_radius_gt.float())
        l_pocket = l_pocket_center + l_pocket_radius

        # bind BCE
        l_bind = F.binary_cross_entropy_with_logits(bind_logit.float(), bind_label.float())

        # affinity only on pos
        if is_pos.any():
            l_aff = F.smooth_l1_loss(aff_pred.float()[is_pos], affinity_gt.float()[is_pos])
        else:
            l_aff = aff_pred.new_tensor(0.0)

    # complementarity + flex
    l_comp = surface_complementarity_loss_pocket(
        rec_tokens, lig_tokens, rec_centers, lig_centers, rec_mask, lig_mask,
        pc_pred, pr_pred, contact_thresh=args.contact_thresh, pocket_extra=args.pocket_extra
    )
    l_flex_rec = local_flexibility_loss(rec_tokens, rec_centers, rec_mask, k_neighbors=args.flex_knn)
    l_flex_lig = local_flexibility_loss(lig_tokens, lig_centers, lig_mask, k_neighbors=args.flex_knn)
    l_flex = 0.5 * (l_flex_rec + l_flex_lig)

    total = (args.w_pocket * l_pocket +
             args.w_bind * l_bind +
             args.w_aff * l_aff +
             args.w_comp * l_comp +
             args.w_flex * l_flex)

    loss_dict = dict(
        total=float(total.detach().item()),
        pocket=float(l_pocket.detach().item()),
        bind=float(l_bind.detach().item()),
        aff=float(l_aff.detach().item()),
        comp=float(l_comp.detach().item()),
        flex=float(l_flex.detach().item()),
    )
    return total, loss_dict


# ============================================================
# 6) Train loop
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch, args, scaler=None):
    model.train()
    iters = len(loader)
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    total = 0.0
    comp_sum = {k: 0.0 for k in ["pocket", "bind", "aff", "comp", "flex"]}

    for it, batch in enumerate(pbar):
        (rec_feats, rec_centers, rec_mask,
         lig_feats, lig_centers, lig_mask,
         pocket_center_gt, pocket_radius_gt,
         bind_label_pos, affinity_gt_pos) = batch

        rec_feats = rec_feats.to(device, non_blocking=True)
        rec_centers = rec_centers.to(device, non_blocking=True)
        rec_mask = rec_mask.to(device, non_blocking=True)

        lig_feats = lig_feats.to(device, non_blocking=True)
        lig_centers = lig_centers.to(device, non_blocking=True)
        lig_mask = lig_mask.to(device, non_blocking=True)

        pocket_center_gt = pocket_center_gt.to(device, non_blocking=True)
        pocket_radius_gt = pocket_radius_gt.to(device, non_blocking=True)
        bind_label_pos = bind_label_pos.to(device, non_blocking=True)
        affinity_gt_pos = affinity_gt_pos.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # negcsv_mode: csv already contains pos/neg, DO NOT create additional negatives
        rec_feats_all, rec_centers_all, rec_mask_all = rec_feats, rec_centers, rec_mask
        lig_feats_all, lig_centers_all, lig_mask_all = lig_feats, lig_centers, lig_mask
        pocket_center_gt_all, pocket_radius_gt_all = pocket_center_gt, pocket_radius_gt
        bind_label_all, affinity_gt_all = bind_label_pos, affinity_gt_pos
        is_pos = (bind_label_pos > 0.5)

        ctx = torch.cuda.amp.autocast(enabled=(scaler is not None))
        with ctx:
            out = model(rec_feats_all, rec_centers_all, rec_mask_all,
                        lig_feats_all, lig_centers_all, lig_mask_all)
            loss, loss_dict = compute_losses(
                out,
                pocket_center_gt_all, pocket_radius_gt_all,
                bind_label_all, affinity_gt_all,
                is_pos,
                args,
            )

        # backward
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # running avg
        total += float(loss.detach().item())
        for k in comp_sum:
            comp_sum[k] += float(loss_dict[k])

        denom = it + 1
        avg_loss = total / denom
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "pocket": f"{comp_sum['pocket']/denom:.3f}",
            "bind": f"{comp_sum['bind']/denom:.3f}",
            "aff": f"{comp_sum['aff']/denom:.3f}",
            "comp": f"{comp_sum['comp']/denom:.3f}",
            "flex": f"{comp_sum['flex']/denom:.3f}",
        })

        # quick NaN guard (stop early to help debug)
        if not np.isfinite(avg_loss):
            tqdm.write(f"[NaN] encountered at epoch={epoch} it={it+1}. Stop.")
            break

    return total / max(1, iters)


def load_pretrained_vqmae_encoders(model: DockingModel, ckpt_path: str,
                                   d_model: int, nhead: int, nlayers: int, K: int, dropout: float,
                                   device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained_vqmae not found: {ckpt_path}")
    print(f"[Stage3] Loading SurfVQMAE encoder from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    vq = SurfVQMAE(
        in_dim=6, d_model=d_model, nhead=nhead,
        nlayers=nlayers, K=K, num_codes=2048, code_dim=d_model, dropout=dropout,
    ).to(device)
    vq.load_state_dict(ckpt["model"], strict=False)

    model.encoder.rec_encoder.local.load_state_dict(vq.local.state_dict())
    model.encoder.rec_encoder.blocks.load_state_dict(vq.blocks.state_dict())
    model.encoder.lig_encoder.local.load_state_dict(vq.local.state_dict())
    model.encoder.lig_encoder.blocks.load_state_dict(vq.blocks.state_dict())
    print("[Stage3] Encoder weights copied into receptor & ligand encoders.")


# ============================================================
# 7) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Stage3 pocket-conditioned docking on PDBbind (xlsx ref + neg-sample csv)")

    # paths (your defaults)
    ap.add_argument("--train_csv", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/stage3_negative_sample_pdbbind.csv",
                    help="Negative-sampled pose CSV (comma-separated).")
    ap.add_argument("--npz_root", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/Processed_pdbbind_per_chain",
                    help="Stage1 output directory with {PDBID}_{CHAIN_GROUP}.npz")

    ap.add_argument("--ref_xlsx", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/PDBbind-CN_v2020_PP_20231108.xlsx",
                    help="PDBbind reference xlsx (for affinity label if aff_source=ref_xlsx).")
    ap.add_argument("--ref_sheet", type=str, default=None,
                    help="Optional sheet name. If None, auto-try sheets.")
    ap.add_argument("--aff_source", type=str, default="pose_csv",
                    choices=["pose_csv", "ref_xlsx"],
                    help="Affinity target source: pose_csv uses dG_bind in train_csv; ref_xlsx uses xlsx by row_idx.")
    ap.add_argument("--ref_aff_col", type=str, default=None,
                    help="If aff_source=ref_xlsx, specify the affinity column name explicitly (optional).")
    ap.add_argument("--aff_clip", type=float, default=200.0,
                    help="Clip affinity target to [-aff_clip, aff_clip] to prevent AMP overflow/NaN.")

    ap.add_argument("--balance_pos_neg", action="store_true",
                    help="Use WeightedRandomSampler to balance pos/neg.")

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--K", type=int, default=50)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)

    # loss weights
    ap.add_argument("--w_pocket", type=float, default=0.005)
    ap.add_argument("--w_bind", type=float, default=20.0)
    ap.add_argument("--w_aff", type=float, default=10.0)
    ap.add_argument("--w_comp", type=float, default=5.0)
    ap.add_argument("--w_flex", type=float, default=20.0)

    # geometry params
    ap.add_argument("--contact_thresh", type=float, default=5.0)
    ap.add_argument("--flex_knn", type=int, default=8)
    ap.add_argument("--pocket_extra", type=float, default=2.0)

    # ckpt
    ap.add_argument("--save_dir", type=str, default="./ckpts_stage3_pocket_conditioned_pdbbind")
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained_vqmae", type=str, default="", help="SurfVQMAE ckpt path to init encoders")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = PDBbindNegSampleCSVDataset(
        train_csv=args.train_csv,
        npz_root=args.npz_root,
        K=args.K,
        seq_len=args.seq_len,
        cache_npz=True,
        interface_cutoff=args.contact_thresh,
        pocket_margin=2.0,
        ref_xlsx=args.ref_xlsx,
        ref_sheet=args.ref_sheet,
        aff_source=args.aff_source,
        ref_aff_col=args.ref_aff_col,
        aff_clip=args.aff_clip,
    )

    sampler = None
    shuffle = True
    if args.balance_pos_neg:
        from torch.utils.data import WeightedRandomSampler
        labels = np.asarray(dataset.bind_labels, dtype=np.int64)  # 1=pos,0=neg
        num_pos = int(labels.sum())
        num_neg = int(len(labels) - num_pos)
        if num_pos > 0 and num_neg > 0:
            w_pos = 0.5 / num_pos
            w_neg = 0.5 / num_neg
            weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=docking_collate_fn,
    )

    model = DockingModel(
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        K=args.K,
        dropout=args.dropout,
    ).to(device)

    if args.pretrained_vqmae:
        load_pretrained_vqmae_encoders(
            model,
            args.pretrained_vqmae,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            K=args.K,
            dropout=args.dropout,
            device=device,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch, args, scaler=scaler)
        print(f"[Epoch {epoch}] avg_loss = {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"e{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"[Stage3] Saved checkpoint: {ckpt_path}")

    final_ckpt = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": vars(args),
    }, final_ckpt)
    print(f"[Stage3] Training finished. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
