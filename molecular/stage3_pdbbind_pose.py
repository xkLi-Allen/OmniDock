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
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    for ch in [",", "+", " ", "_", ":", "|", "/", "\\", ";"]:
        s = s.replace(ch, "")
    s = s.upper()
    s = "".join([c for c in s if c.isalnum()])
    s = "".join(sorted(set(list(s))))
    return s

def _is_finite_number(x) -> bool:
    try:
        v = float(x)
        return np.isfinite(v)
    except Exception:
        return False

def _parse_float_list(s: str, n: int) -> np.ndarray:
    arr = np.fromstring(str(s), sep=",", dtype=np.float32)
    if arr.size != n:
        raise ValueError(f"Expected {n} floats, got {arr.size}: {str(s)[:120]}")
    return arr


# ============================================================
# 1) PDBbind xlsx loader (optional for ref label)
# ============================================================

_PDB_CODE_COL_PATTERNS = [
    r"\bpdb\s*code\b", r"\bpdb\s*id\b", r"\bpdb\b", r"pdb编号", r"pdb\s*码"
]
_AFF_COL_PATTERNS = [
    r"delta\s*g", r"Δg", r"\bdg\b", r"binding\s*free", r"binding\s*energy",
    r"\bkd\b", r"\bki\b", r"ic50", r"affinity", r"亲和", r"解离", r"结合自由能"
]

def _score_header_row(row_vals: List[str]) -> float:
    nonempty = [v for v in row_vals if v and v.lower() != "nan"]
    if len(nonempty) < 3:
        return -1e9
    joined = " | ".join(nonempty).lower()
    score = 0.0
    if "pdb" in joined: score += 3.0
    if "code" in joined or "id" in joined or "编号" in joined: score += 2.0
    if "chain" in joined or "链" in joined: score += 1.0
    for pat in _AFF_COL_PATTERNS:
        if re.search(pat, joined, flags=re.IGNORECASE):
            score += 0.7
            break
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
    for c in cols:
        s = df[c].astype(str).str.strip()
        hit = s.str.match(r"^[0-9][A-Za-z0-9]{3}$").mean()
        if hit > 0.5:
            return c
    return None

def load_pdbbind_xlsx(xlsx_path: str, sheet_name: Optional[str] = None, max_scan_rows: int = 80) -> pd.DataFrame:
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
            best_i, best_score = None, -1e18
            scan_n = min(max_scan_rows, raw.shape[0])
            for i in range(scan_n):
                row_vals = [str(x).strip() for x in raw.iloc[i].tolist()]
                sc = _score_header_row(row_vals)
                if sc > best_score:
                    best_score, best_i = sc, i
            if best_i is None:
                continue
            header = _make_unique_columns([str(x).strip() for x in raw.iloc[best_i].tolist()])
            df = raw.iloc[best_i + 1:].copy()
            df.columns = header
            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            pdb_col = _find_pdb_col(df)
            if pdb_col is None:
                continue
            return df
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise ValueError(f"Failed to load usable sheet from xlsx: {xlsx_path}. last_err={last_err}")
    raise ValueError(f"Cannot detect usable table from xlsx: {xlsx_path}.")


# ============================================================
# 2) Dataset: GROUPED pose-csv (1 pos + N neg per complex)
# ============================================================

class PDBbindPoseGroupDataset(Dataset):
    """
    pose-csv columns expected:
      row_idx,pdb_str,pdb_id,rec_chains,lig_chains,pose_type,neg_id,R_flat,shift[,dG_bind]

    Each __getitem__ returns one "group" (one complex):
      - 1 POS pose + (num_neg) NEG poses
      - pocket_center_gt/radius_gt computed once from native POS geometry (pose-invariant target)
      - receptor sampling is pocket-centered (stable interface)
    """

    def __init__(
        self,
        pose_csv: str,
        npz_root: str,
        K: int = 50,
        seq_len: int = 512,
        cache_npz: bool = True,
        interface_cutoff: float = 8.0,
        pocket_margin: float = 2.0,
        # affinity label
        ref_xlsx: Optional[str] = None,
        ref_sheet: Optional[str] = None,
        aff_source: str = "pose_csv",      # pose_csv | ref_xlsx
        ref_aff_col: Optional[str] = None,
        aff_clip: float = 200.0,
        # grouping / sampling
        num_neg: int = 5,
        group_key_mode: str = "row_idx",   # row_idx | pdb_chain
        lig_sampling: str = "order",       # order | com
        deterministic: bool = False,
    ):
        super().__init__()
        assert os.path.isfile(pose_csv), f"pose_csv not found: {pose_csv}"
        assert os.path.isdir(npz_root), f"npz_root not found: {npz_root}"
        assert group_key_mode in ["row_idx", "pdb_chain"]
        assert lig_sampling in ["order", "com"]

        self.pose_csv = pose_csv
        self.npz_root = npz_root
        self.K = int(K)
        self.seq_len = int(seq_len)
        self.cache_npz = bool(cache_npz)
        self.interface_cutoff = float(interface_cutoff)
        self.pocket_margin = float(pocket_margin)
        self.aff_source = aff_source
        self.aff_clip = float(aff_clip)
        self.num_neg = int(num_neg)
        self.group_key_mode = group_key_mode
        self.lig_sampling = lig_sampling
        self.deterministic = bool(deterministic)

        df_pose = pd.read_csv(pose_csv)
        req = {"row_idx", "pose_type", "R_flat", "shift", "pdb_id", "rec_chains", "lig_chains"}
        miss = req - set(df_pose.columns)
        if miss:
            raise ValueError(f"pose_csv missing columns: {sorted(list(miss))}")
        self.df_pose = df_pose.reset_index(drop=True)

        # optional ref affinity from xlsx
        self.aff_by_rowidx = None
        self._ref_mode_ok = False
        if self.aff_source == "ref_xlsx":
            assert ref_xlsx is not None and os.path.isfile(ref_xlsx), f"ref_xlsx not found: {ref_xlsx}"
            df_ref = load_pdbbind_xlsx(ref_xlsx, sheet_name=ref_sheet).reset_index(drop=True)
            pdb_col = _find_pdb_col(df_ref)
            if pdb_col is None:
                raise ValueError(f"Cannot find PDB column in xlsx. cols={df_ref.columns.tolist()}")

            if ref_aff_col is not None and ref_aff_col in df_ref.columns:
                aff_col = ref_aff_col
            else:
                aff_col = None
                for c in df_ref.columns:
                    c0 = str(c).strip().lower()
                    if any(re.search(p, c0, flags=re.IGNORECASE) for p in _AFF_COL_PATTERNS):
                        aff_col = c
                        break
                if aff_col is None:
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
                raise ValueError(f"Cannot auto-detect affinity column in xlsx, pass --ref_aff_col.")
            x = pd.to_numeric(df_ref[aff_col], errors="coerce").fillna(0.0).astype(np.float32).values
            x = np.clip(x, -self.aff_clip, self.aff_clip)
            self.aff_by_rowidx = x
            self._ref_mode_ok = True
            print(f"[Dataset] ref_xlsx loaded. aff_col={aff_col}, rows={len(x)}")

        self._npz_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # ---------- build groups ----------
        # Normalize ids & chains
        norm_rows = []
        skipped = 0
        for i, row in self.df_pose.iterrows():
            pdb_id = str(row["pdb_id"]).strip()[:4].upper()
            rec_ch = _clean_chain_group(str(row["rec_chains"]))
            lig_ch = _clean_chain_group(str(row["lig_chains"]))
            pose_type = str(row["pose_type"]).lower().strip()
            if not pdb_id or not rec_ch or not lig_ch or pose_type not in ["pos", "neg"]:
                skipped += 1
                continue
            rec_npz = self._find_npz(pdb_id, rec_ch)
            lig_npz = self._find_npz(pdb_id, lig_ch)
            if rec_npz is None or lig_npz is None:
                skipped += 1
                continue

            if self.group_key_mode == "row_idx":
                gkey = f"{int(row['row_idx'])}|{pdb_id}|{rec_ch}|{lig_ch}"
            else:
                gkey = f"{pdb_id}|{rec_ch}|{lig_ch}"

            norm_rows.append((i, gkey, rec_npz, lig_npz, pose_type))

        if not norm_rows:
            raise RuntimeError("No usable rows found after checking npz existence.")

        # group -> indices
        groups: Dict[str, Dict[str, Any]] = {}
        for i, gkey, rec_npz, lig_npz, pose_type in norm_rows:
            if gkey not in groups:
                groups[gkey] = {
                    "rows": [],
                    "pos": [],
                    "neg": [],
                    "rec_npz": rec_npz,
                    "lig_npz": lig_npz,
                }
            groups[gkey]["rows"].append(i)
            if pose_type == "pos":
                groups[gkey]["pos"].append(i)
            else:
                groups[gkey]["neg"].append(i)

        # keep only groups having pos+neg
        self.group_keys: List[str] = []
        self.group_info: Dict[str, Dict[str, Any]] = {}
        pocket_cache = {}

        for gkey, info in groups.items():
            if len(info["pos"]) == 0 or len(info["neg"]) == 0:
                continue

            # precompute pocket GT once per group from native geometry (POS, no transform)
            rec = self._load_surface(info["rec_npz"])
            lig = self._load_surface(info["lig_npz"])
            pocket_center, pocket_radius = self._compute_pocket_from_interface(rec["centers"], lig["centers"])

            pocket_cache[gkey] = (pocket_center, pocket_radius)
            info["pocket_center_gt"] = pocket_center
            info["pocket_radius_gt"] = pocket_radius
            self.group_keys.append(gkey)
            self.group_info[gkey] = info

        print(f"[PoseGroupDataset] pose_rows={len(self.df_pose)} usable_groups={len(self.group_keys)} skipped_rows={skipped}")
        if len(self.group_keys) == 0:
            raise RuntimeError("No usable groups (need >=1 pos and >=1 neg per group).")

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

    def _compute_pocket_from_interface(
        self,
        rec_centers_all: np.ndarray,
        lig_centers_all: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pseudo pocket from interface contacts on the *native* pose.
        NOTE: This still does O(Nr*Nl) distance, but only once per group.
        """
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

    def _sample_rec_pocket_centered(
        self,
        xs: np.ndarray, ns: np.ndarray, centers: np.ndarray, knn: np.ndarray,
        pocket_center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic: choose seq_len receptor patches closest to pocket_center.
        Return feats (T,K,6), ctrs (T,3), mask (T,) True=PAD.
        """
        Nc = centers.shape[0]
        if Nc == 0:
            T = self.seq_len
            feats = np.zeros((T, self.K, 6), dtype=np.float32)
            ctrs = np.zeros((T, 3), dtype=np.float32)
            mask = np.ones((T,), dtype=np.bool_)
            return feats, ctrs, mask

        d = np.linalg.norm(centers - pocket_center.reshape(1, 3), axis=-1)
        idx = np.argsort(d)  # closest first
        if Nc >= self.seq_len:
            sel = idx[:self.seq_len]
            mask = np.zeros((self.seq_len,), dtype=np.bool_)
        else:
            # pad by repeating last
            pad_n = self.seq_len - Nc
            sel = np.concatenate([idx, np.full((pad_n,), idx[-1], dtype=idx.dtype)], axis=0)
            mask = np.concatenate([np.zeros((Nc,), dtype=np.bool_), np.ones((pad_n,), dtype=np.bool_)], axis=0)

        pts_idx = knn[sel]                 # (T,K)
        ctrs = centers[sel]                # (T,3)
        rel_xyz = xs[pts_idx] - ctrs[:, None, :]
        norms = ns[pts_idx]
        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
        return feats, ctrs.astype(np.float32), mask

    def _sample_lig(
        self,
        xs: np.ndarray, ns: np.ndarray, centers: np.ndarray, knn: np.ndarray, order: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ligand sampling:
          - "order": deterministic take order[:seq_len]
          - "com": take patches closest to ligand COM (stable, translation-invariant selection)
        """
        Nc = centers.shape[0]
        if Nc == 0:
            T = self.seq_len
            feats = np.zeros((T, self.K, 6), dtype=np.float32)
            ctrs = np.zeros((T, 3), dtype=np.float32)
            mask = np.ones((T,), dtype=np.bool_)
            return feats, ctrs, mask

        if Nc >= self.seq_len:
            if self.lig_sampling == "order":
                sel = order[:self.seq_len]
            else:
                com = centers.mean(axis=0, keepdims=True)
                d = np.linalg.norm(centers - com, axis=-1)
                sel = np.argsort(d)[:self.seq_len]
            mask = np.zeros((self.seq_len,), dtype=np.bool_)
        else:
            pad_n = self.seq_len - Nc
            if self.lig_sampling == "order":
                sel0 = order
            else:
                com = centers.mean(axis=0, keepdims=True)
                d = np.linalg.norm(centers - com, axis=-1)
                sel0 = np.argsort(d)
            sel = np.concatenate([sel0, np.full((pad_n,), sel0[-1], dtype=sel0.dtype)], axis=0)
            mask = np.concatenate([np.zeros((Nc,), dtype=np.bool_), np.ones((pad_n,), dtype=np.bool_)], axis=0)

        pts_idx = knn[sel]                 # (T,K)
        ctrs = centers[sel]                # (T,3)
        rel_xyz = xs[pts_idx] - ctrs[:, None, :]
        norms = ns[pts_idx]
        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
        return feats, ctrs.astype(np.float32), mask

    def __len__(self) -> int:
        return len(self.group_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gkey = self.group_keys[idx]
        info = self.group_info[gkey]
        rec_npz = info["rec_npz"]
        lig_npz = info["lig_npz"]

        # choose 1 pos + num_neg negs
        pos_i = info["pos"][0] if self.deterministic else np.random.choice(info["pos"])
        neg_pool = info["neg"]
        if len(neg_pool) >= self.num_neg:
            neg_is = neg_pool[:self.num_neg] if self.deterministic else list(np.random.choice(neg_pool, size=self.num_neg, replace=False))
        else:
            neg_is = neg_pool if self.deterministic else list(np.random.choice(neg_pool, size=self.num_neg, replace=True))

        pose_indices = [pos_i] + list(neg_is)  # length P=1+num_neg
        P = len(pose_indices)

        rec = self._load_surface(rec_npz)
        lig = self._load_surface(lig_npz)

        # pocket GT (pose-invariant)
        pocket_center_gt = info["pocket_center_gt"]  # (3,)
        pocket_radius_gt = info["pocket_radius_gt"]  # (1,)

        # sample receptor ONCE per group (stable pocket-centered)
        rec_feats, rec_centers, rec_mask = self._sample_rec_pocket_centered(
            rec["xs"], rec["ns"], rec["centers"], rec["knn"], pocket_center_gt
        )

        # per-pose ligand transform + sampling
        lig_feats_all = []
        lig_centers_all = []
        lig_mask_all = []
        bind_label_all = []
        affinity_all = []
        is_pos_all = []
        name_all = []

        for pi in pose_indices:
            row = self.df_pose.iloc[pi]
            pose_type = str(row["pose_type"]).lower().strip()
            is_pos = 1.0 if pose_type == "pos" else 0.0

            # sample ligand (on native centers first)
            lf, lc, lm = self._sample_lig(lig["xs"], lig["ns"], lig["centers"], lig["knn"], lig["order"])

            # apply stored transform for NEG poses
            if is_pos < 0.5:
                R = _parse_float_list(row["R_flat"], 9).reshape(3, 3)
                shift = _parse_float_list(row["shift"], 3).reshape(3)

                # transform centers (global) around COM of sampled window
                com = lc.mean(axis=0, keepdims=True).astype(np.float32)
                relc = lc - com
                lc = relc @ R.T + com + shift[None, :]

                # rotate local rel_xyz & normals
                rel = lf[..., :3]
                nrm = lf[..., 3:]
                rel = rel @ R.T
                nrm = nrm @ R.T
                lf = np.concatenate([rel, nrm], axis=-1).astype(np.float32)

            lig_feats_all.append(torch.from_numpy(lf))
            lig_centers_all.append(torch.from_numpy(lc))
            lig_mask_all.append(torch.from_numpy(lm.astype(np.bool_)))

            bind_label_all.append(float(is_pos))
            is_pos_all.append(bool(is_pos > 0.5))

            # affinity: POS-only
            aff = 0.0
            if is_pos > 0.5:
                if self.aff_source == "pose_csv" and "dG_bind" in row.index:
                    v = row["dG_bind"]
                    aff = float(v) if _is_finite_number(v) else 0.0
                elif self.aff_source == "ref_xlsx" and self._ref_mode_ok and self.aff_by_rowidx is not None:
                    ridx = int(row["row_idx"])
                    if 0 <= ridx < len(self.aff_by_rowidx):
                        aff = float(self.aff_by_rowidx[ridx])
            aff = float(np.clip(aff, -self.aff_clip, self.aff_clip))
            affinity_all.append(aff if is_pos > 0.5 else 0.0)

            neg_id = int(row["neg_id"]) if "neg_id" in row.index and _is_finite_number(row["neg_id"]) else -1
            pdb_str = str(row.get("pdb_str", f"{row['pdb_id']}_{row['rec_chains']}_{row['lig_chains']}"))
            name_all.append(f"{pdb_str}|{pose_type}|{neg_id}")

        # pack to tensors
        rec_feats_t = torch.from_numpy(rec_feats)          # (Tr,K,6)
        rec_centers_t = torch.from_numpy(rec_centers)      # (Tr,3)
        rec_mask_t = torch.from_numpy(rec_mask.astype(np.bool_))  # (Tr,)

        lig_feats_t = torch.stack(lig_feats_all, dim=0)    # (P,Tl,K,6)
        lig_centers_t = torch.stack(lig_centers_all, dim=0)# (P,Tl,3)
        lig_mask_t = torch.stack(lig_mask_all, dim=0)      # (P,Tl)

        bind_label_t = torch.tensor(bind_label_all, dtype=torch.float32)  # (P,)
        affinity_t = torch.tensor(affinity_all, dtype=torch.float32)      # (P,)
        is_pos_t = torch.tensor(is_pos_all, dtype=torch.bool)             # (P,)

        pocket_center_t = torch.from_numpy(pocket_center_gt.astype(np.float32))  # (3,)
        pocket_radius_t = torch.from_numpy(pocket_radius_gt.astype(np.float32))  # (1,)

        return {
            "rec_feats": rec_feats_t,
            "rec_centers": rec_centers_t,
            "rec_mask": rec_mask_t,
            "lig_feats": lig_feats_t,
            "lig_centers": lig_centers_t,
            "lig_mask": lig_mask_t,
            "pocket_center_gt": pocket_center_t,
            "pocket_radius_gt": pocket_radius_t,
            "bind_label": bind_label_t,
            "affinity": affinity_t,
            "is_pos": is_pos_t,
            "names": name_all,
            "group_key": gkey,
        }


def group_collate_fn(batch: List[Dict[str, Any]]):
    """
    Batch:
      rec_*: (Tr,...) per item
      lig_*: (P,Tl,...) per item
    We pad Tr/Tl to max within batch (Tr and Tl should usually be seq_len already).
    """
    B = len(batch)
    P = batch[0]["lig_feats"].shape[0]
    K = batch[0]["rec_feats"].shape[1]

    Tr_max = max(b["rec_feats"].shape[0] for b in batch)
    Tl_max = max(b["lig_feats"].shape[1] for b in batch)

    rec_feats_list, rec_centers_list, rec_mask_list = [], [], []
    lig_feats_list, lig_centers_list, lig_mask_list = [], [], []
    pocket_centers, pocket_radii = [], []
    bind_labels, affinities, is_pos_list = [], [], []
    names_list, gkeys = [], []

    for b in batch:
        # rec pad
        rf = b["rec_feats"]; rc = b["rec_centers"]; rm = b["rec_mask"]
        Tr = rf.shape[0]
        if Tr < Tr_max:
            pad = Tr_max - Tr
            rf = torch.cat([rf, torch.zeros((pad, K, 6), dtype=rf.dtype)], dim=0)
            rc = torch.cat([rc, torch.zeros((pad, 3), dtype=rc.dtype)], dim=0)
            rm = torch.cat([rm, torch.ones((pad,), dtype=torch.bool)], dim=0)
        rec_feats_list.append(rf); rec_centers_list.append(rc); rec_mask_list.append(rm)

        # lig pad (pad on Tl dimension)
        lf = b["lig_feats"]; lc = b["lig_centers"]; lm = b["lig_mask"]  # (P,Tl,...)
        Tl = lf.shape[1]
        if Tl < Tl_max:
            pad = Tl_max - Tl
            lf = torch.cat([lf, torch.zeros((P, pad, K, 6), dtype=lf.dtype)], dim=1)
            lc = torch.cat([lc, torch.zeros((P, pad, 3), dtype=lc.dtype)], dim=1)
            lm = torch.cat([lm, torch.ones((P, pad), dtype=torch.bool)], dim=1)
        lig_feats_list.append(lf); lig_centers_list.append(lc); lig_mask_list.append(lm)

        pocket_centers.append(b["pocket_center_gt"])
        pocket_radii.append(b["pocket_radius_gt"])
        bind_labels.append(b["bind_label"])
        affinities.append(b["affinity"])
        is_pos_list.append(b["is_pos"])
        names_list.append(b["names"])
        gkeys.append(b["group_key"])

    rec_feats = torch.stack(rec_feats_list, dim=0)      # (B,Tr,K,6)
    rec_centers = torch.stack(rec_centers_list, dim=0)  # (B,Tr,3)
    rec_mask = torch.stack(rec_mask_list, dim=0)        # (B,Tr)

    lig_feats = torch.stack(lig_feats_list, dim=0)      # (B,P,Tl,K,6)
    lig_centers = torch.stack(lig_centers_list, dim=0)  # (B,P,Tl,3)
    lig_mask = torch.stack(lig_mask_list, dim=0)        # (B,P,Tl)

    pocket_center_gt = torch.stack(pocket_centers, dim=0)  # (B,3)
    pocket_radius_gt = torch.stack(pocket_radii, dim=0)    # (B,1)

    bind_label = torch.stack(bind_labels, dim=0)           # (B,P)
    affinity = torch.stack(affinities, dim=0)              # (B,P)
    is_pos = torch.stack(is_pos_list, dim=0)               # (B,P)

    return (rec_feats, rec_centers, rec_mask,
            lig_feats, lig_centers, lig_mask,
            pocket_center_gt, pocket_radius_gt,
            bind_label, affinity, is_pos,
            names_list, gkeys)


# ============================================================
# 3) Model: distance-biased cross attention + contact-map supervision
# ============================================================

from unsupervised_pre_training import (  # type: ignore
    PointMLP, SurfFormerBlock, SurfVQMAE
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

class RBFDistanceBias(nn.Module):
    """
    dist (B, Lq, Lk) -> bias (B, H, Lq, Lk)
    """
    def __init__(self, nhead: int, num_rbf: int = 16, rbf_min: float = 0.0, rbf_max: float = 30.0):
        super().__init__()
        assert num_rbf >= 2
        self.nhead = nhead
        self.num_rbf = num_rbf
        mus = torch.linspace(rbf_min, rbf_max, num_rbf)
        delta = (rbf_max - rbf_min) / (num_rbf - 1)
        gamma = 1.0 / max(delta * delta, 1e-6)
        self.register_buffer("mus", mus)
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.proj = nn.Linear(num_rbf, nhead, bias=False)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        x = dist.unsqueeze(-1)
        rbf = torch.exp(-self.gamma * (x - self.mus) ** 2)
        bias = self.proj(rbf)  # (B,Lq,Lk,H)
        bias = bias.permute(0, 3, 1, 2).contiguous()
        return bias

class CrossAttentionWithDistBias(nn.Module):
    """
    logits = (QK^T)/sqrt(d) + bias_scale * bias(dist)
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1,
                 num_rbf=16, rbf_min=0.0, rbf_max=30.0, bias_scale=2.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.bias_scale = float(bias_scale)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.rbf_bias = RBFDistanceBias(nhead=nhead, num_rbf=num_rbf, rbf_min=rbf_min, rbf_max=rbf_max)
        self.drop = nn.Dropout(dropout)

    def forward(self, rec_tokens, lig_tokens, rec_centers, lig_centers, lig_mask=None):
        B, Tr, _ = rec_tokens.shape
        _, Tl, _ = lig_tokens.shape

        q = self.q_proj(rec_tokens).view(B, Tr, self.nhead, self.d_head).transpose(1, 2)  # (B,H,Tr,d)
        k = self.k_proj(lig_tokens).view(B, Tl, self.nhead, self.d_head).transpose(1, 2)  # (B,H,Tl,d)
        v = self.v_proj(lig_tokens).view(B, Tl, self.nhead, self.d_head).transpose(1, 2)  # (B,H,Tl,d)

        with torch.cuda.amp.autocast(enabled=False):
            dist = torch.cdist(rec_centers.float(), lig_centers.float())  # (B,Tr,Tl)
            bias = self.rbf_bias(dist)  # (B,H,Tr,Tl)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.bias_scale * bias.to(attn_logits.dtype)

        if lig_mask is not None:
            mask = lig_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,Tl) True=PAD
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        with torch.cuda.amp.autocast(enabled=False):
            attn = torch.softmax(attn_logits.float(), dim=-1).to(attn_logits.dtype)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # (B,H,Tr,d)
        out = out.transpose(1, 2).contiguous().view(B, Tr, self.d_model)
        out = self.out_proj(out)
        return out, attn

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
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1,
                 num_rbf=16, rbf_min=0.0, rbf_max=30.0, bias_scale=2.0):
        super().__init__()
        self.rec_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers, K=K, dropout=dropout)
        self.lig_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers, K=K, dropout=dropout)

        self.cross_attn = CrossAttentionWithDistBias(
            d_model=d_model, nhead=nhead, dropout=dropout,
            num_rbf=num_rbf, rbf_min=rbf_min, rbf_max=rbf_max, bias_scale=bias_scale
        )
        self.cross_norm = nn.LayerNorm(d_model)

    def forward(self, rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask):
        rec_tokens = self.rec_encoder(rec_feats, rec_centers)
        lig_tokens = self.lig_encoder(lig_feats, lig_centers)

        cross, attn = self.cross_attn(
            rec_tokens, lig_tokens,
            rec_centers=rec_centers, lig_centers=lig_centers,
            lig_mask=lig_mask
        )
        cross = self.cross_norm(cross)
        pair_repr = cross.mean(dim=1)
        return {
            "rec_tokens": rec_tokens,
            "lig_tokens": lig_tokens,
            "cross_tokens": cross,
            "pair_repr": pair_repr,
            "cross_attn": attn,  # (B,H,Tr,Tl)
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
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1,
                 num_rbf=16, rbf_min=0.0, rbf_max=30.0, bias_scale=2.0):
        super().__init__()
        self.encoder = PairEncoder(
            d_model=d_model, nhead=nhead, nlayers=nlayers, K=K, dropout=dropout,
            num_rbf=num_rbf, rbf_min=rbf_min, rbf_max=rbf_max, bias_scale=bias_scale
        )
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
            "cross_attn": enc_out["cross_attn"],
            "rec_centers": rec_centers,
            "lig_centers": lig_centers,
            "rec_mask": rec_mask,
            "lig_mask": lig_mask,
        }


# ============================================================
# 4) Losses (base + ranking)
# ============================================================

def attn_contact_loss_pos_only(
    attn: torch.Tensor,               # (B,H,Tr,Tl)
    rec_centers: torch.Tensor,        # (B,Tr,3)
    lig_centers: torch.Tensor,        # (B,Tl,3)
    rec_mask: torch.Tensor,           # (B,Tr) True=PAD
    lig_mask: torch.Tensor,           # (B,Tl) True=PAD
    bind_label: torch.Tensor,         # (B,) 1=pos
    contact_thresh: float = 5.0,
    eps: float = 1e-8,
):
    B, H, Tr, Tl = attn.shape
    attn_avg = attn.mean(dim=1)  # (B,Tr,Tl)
    total = attn.new_tensor(0.0)
    used = 0

    for b in range(B):
        if float(bind_label[b].item()) < 0.5:
            continue
        vr = ~rec_mask[b]
        vl = ~lig_mask[b]
        if vr.sum().item() == 0 or vl.sum().item() == 0:
            continue

        rc = rec_centers[b][vr]
        lc = lig_centers[b][vl]
        a = attn_avg[b][vr][:, vl]

        with torch.cuda.amp.autocast(enabled=False):
            dist = torch.cdist(rc.float(), lc.float())
            contact = (dist <= float(contact_thresh)).float()

            row_sum = contact.sum(dim=-1, keepdim=True)
            no_contact = (row_sum <= 0.0)
            if no_contact.any():
                contact = contact.clone()
                contact[no_contact.squeeze(-1)] = 1.0
                row_sum = contact.sum(dim=-1, keepdim=True)

            target = contact / row_sum.clamp(min=1.0)
            p = a.float().clamp(min=eps)
            loss = -(target * torch.log(p)).sum(dim=-1).mean()

        total = total + loss.to(total.dtype)
        used += 1

    if used == 0:
        return attn.new_tensor(0.0)
    return total / used


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


def base_losses_per_pose(
    model_out,
    pocket_center_gt, pocket_radius_gt,
    bind_label, affinity_gt,
    is_pos: torch.Tensor,
    args
):
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
    cross_attn = model_out.get("cross_attn", None)

    with torch.cuda.amp.autocast(enabled=False):
        l_pocket_center = F.mse_loss(pc_pred.float(), pocket_center_gt.float())
        l_pocket_radius = F.mse_loss(pr_pred.float(), pocket_radius_gt.float())
        l_pocket = l_pocket_center + l_pocket_radius

        l_bind = F.binary_cross_entropy_with_logits(bind_logit.float(), bind_label.float())

        if is_pos.any():
            l_aff = F.smooth_l1_loss(aff_pred.float()[is_pos], affinity_gt.float()[is_pos])
        else:
            l_aff = aff_pred.new_tensor(0.0)

    l_comp = surface_complementarity_loss_pocket(
        rec_tokens, lig_tokens, rec_centers, lig_centers, rec_mask, lig_mask,
        pc_pred, pr_pred, contact_thresh=args.contact_thresh, pocket_extra=args.pocket_extra
    )
    l_flex_rec = local_flexibility_loss(rec_tokens, rec_centers, rec_mask, k_neighbors=args.flex_knn)
    l_flex_lig = local_flexibility_loss(lig_tokens, lig_centers, lig_mask, k_neighbors=args.flex_knn)
    l_flex = 0.5 * (l_flex_rec + l_flex_lig)

    if cross_attn is not None and args.w_attn > 0:
        l_attn = attn_contact_loss_pos_only(
            cross_attn, rec_centers, lig_centers, rec_mask, lig_mask,
            bind_label=bind_label, contact_thresh=args.contact_thresh
        )
    else:
        l_attn = rec_tokens.new_tensor(0.0)

    base_total = (args.w_pocket * l_pocket +
                  args.w_bind * l_bind +
                  args.w_aff * l_aff +
                  args.w_comp * l_comp +
                  args.w_flex * l_flex +
                  args.w_attn * l_attn)

    loss_dict = dict(
        pocket=float(l_pocket.detach().item()),
        bind=float(l_bind.detach().item()),
        aff=float(l_aff.detach().item()),
        comp=float(l_comp.detach().item()),
        flex=float(l_flex.detach().item()),
        attn=float(l_attn.detach().item()),
        base_total=float(base_total.detach().item()),
    )
    return base_total, loss_dict


def group_ranking_loss(bind_logits_group: torch.Tensor, is_pos_group: torch.Tensor,
                       margin: float = 1.0) -> torch.Tensor:
    """
    bind_logits_group: (B,P)
    is_pos_group:      (B,P) bool

    For each group b:
      s_pos = max score among positives
      hinge = mean relu(margin - (s_pos - s_neg))
      listwise = logsumexp([s_pos] + s_negs) - s_pos
    """
    B, P = bind_logits_group.shape
    total = bind_logits_group.new_tensor(0.0)
    used = 0

    for b in range(B):
        pos_mask = is_pos_group[b]
        neg_mask = ~pos_mask
        if pos_mask.sum().item() == 0 or neg_mask.sum().item() == 0:
            continue

        s_pos = bind_logits_group[b][pos_mask].max()
        s_negs = bind_logits_group[b][neg_mask]

        hinge = F.relu(margin - (s_pos - s_negs)).mean()
        listwise = torch.logsumexp(torch.cat([s_pos.view(1), s_negs.view(-1)], dim=0), dim=0) - s_pos

        total = total + (hinge + listwise)
        used += 1

    if used == 0:
        return bind_logits_group.new_tensor(0.0)
    return total / used


# ============================================================
# 5) Train loop (GROUPED)
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch, args, scaler=None):
    model.train()
    iters = len(loader)
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    total = 0.0
    comp_sum = {k: 0.0 for k in ["base_total", "rank", "pocket", "bind", "aff", "comp", "flex", "attn"]}

    for it, batch in enumerate(pbar):
        (rec_feats, rec_centers, rec_mask,
         lig_feats, lig_centers, lig_mask,
         pocket_center_gt, pocket_radius_gt,
         bind_label, affinity_gt, is_pos,
         names_list, gkeys) = batch

        # shapes:
        # rec_feats: (B,Tr,K,6)
        # lig_feats: (B,P,Tl,K,6)
        B = rec_feats.size(0)
        P = lig_feats.size(1)

        rec_feats = rec_feats.to(device, non_blocking=True)
        rec_centers = rec_centers.to(device, non_blocking=True)
        rec_mask = rec_mask.to(device, non_blocking=True)

        lig_feats = lig_feats.to(device, non_blocking=True)
        lig_centers = lig_centers.to(device, non_blocking=True)
        lig_mask = lig_mask.to(device, non_blocking=True)

        pocket_center_gt = pocket_center_gt.to(device, non_blocking=True)
        pocket_radius_gt = pocket_radius_gt.to(device, non_blocking=True)

        bind_label = bind_label.to(device, non_blocking=True)    # (B,P)
        affinity_gt = affinity_gt.to(device, non_blocking=True)  # (B,P)
        is_pos = is_pos.to(device, non_blocking=True)            # (B,P)

        optimizer.zero_grad(set_to_none=True)

        # flatten poses: treat each pose as one item in forward
        # rec repeats P times
        rec_feats_f = rec_feats.unsqueeze(1).expand(B, P, *rec_feats.shape[1:]).contiguous().view(B * P, *rec_feats.shape[1:])
        rec_centers_f = rec_centers.unsqueeze(1).expand(B, P, *rec_centers.shape[1:]).contiguous().view(B * P, *rec_centers.shape[1:])
        rec_mask_f = rec_mask.unsqueeze(1).expand(B, P, *rec_mask.shape[1:]).contiguous().view(B * P, *rec_mask.shape[1:])

        lig_feats_f = lig_feats.contiguous().view(B * P, *lig_feats.shape[2:])
        lig_centers_f = lig_centers.contiguous().view(B * P, *lig_centers.shape[2:])
        lig_mask_f = lig_mask.contiguous().view(B * P, *lig_mask.shape[2:])

        # pocket GT repeats P times (pose-invariant)
        pocket_center_f = pocket_center_gt.unsqueeze(1).expand(B, P, 3).contiguous().view(B * P, 3)
        pocket_radius_f = pocket_radius_gt.unsqueeze(1).expand(B, P, 1).contiguous().view(B * P, 1)

        bind_label_f = bind_label.contiguous().view(B * P)
        affinity_f = affinity_gt.contiguous().view(B * P)
        is_pos_f = is_pos.contiguous().view(B * P)

        ctx = torch.cuda.amp.autocast(enabled=(scaler is not None))
        with ctx:
            out = model(rec_feats_f, rec_centers_f, rec_mask_f,
                        lig_feats_f, lig_centers_f, lig_mask_f)

            base_total, base_dict = base_losses_per_pose(
                out,
                pocket_center_f, pocket_radius_f,
                bind_label_f, affinity_f,
                is_pos_f,
                args
            )

            # group ranking loss on bind_logit
            bind_logit_f = out["bind_logit"].view(B, P)
            rank_loss = group_ranking_loss(bind_logit_f, is_pos, margin=args.rank_margin)

            loss = base_total + args.w_rank * rank_loss

        if not torch.isfinite(loss).item():
            print("\n[WARN] Non-finite loss. Skipping step.")
            print("example names:", names_list[0] if names_list else None)
            continue

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

        total += float(loss.detach().item())
        comp_sum["base_total"] += float(base_dict["base_total"])
        comp_sum["rank"] += float(rank_loss.detach().item())
        for k in ["pocket", "bind", "aff", "comp", "flex", "attn"]:
            comp_sum[k] += float(base_dict[k])

        denom = it + 1
        pbar.set_postfix({
            "loss": f"{total/denom:.4f}",
            "rank": f"{comp_sum['rank']/denom:.3f}",
            "bind": f"{comp_sum['bind']/denom:.3f}",
            "attn": f"{comp_sum['attn']/denom:.3f}",
            "comp": f"{comp_sum['comp']/denom:.3f}",
        })

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
# 6) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Stage3 pose-ranking improved training (grouped pose_csv)")

    # paths (your defaults)
    ap.add_argument("--pose_csv", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/stage3_negative_sample_pdbbind.csv")
    ap.add_argument("--npz_root", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/Processed_pdbbind_per_chain")

    ap.add_argument("--ref_xlsx", type=str,
                    default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/PDBbind-CN_v2020_PP_20231108.xlsx")
    ap.add_argument("--ref_sheet", type=str, default=None)
    ap.add_argument("--aff_source", type=str, default="pose_csv", choices=["pose_csv", "ref_xlsx"])
    ap.add_argument("--ref_aff_col", type=str, default=None)
    ap.add_argument("--aff_clip", type=float, default=200.0)

    # group settings
    ap.add_argument("--num_neg", type=int, default=5, help="per group: 1 POS + num_neg NEG")
    ap.add_argument("--group_key_mode", type=str, default="row_idx", choices=["row_idx", "pdb_chain"])
    ap.add_argument("--lig_sampling", type=str, default="order", choices=["order", "com"])
    ap.add_argument("--deterministic_group_sampling", action="store_true")

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

    # loss weights (base)
    ap.add_argument("--w_pocket", type=float, default=0.005)
    ap.add_argument("--w_bind", type=float, default=10.0)   # usually can be smaller now
    ap.add_argument("--w_aff", type=float, default=10.0)
    ap.add_argument("--w_comp", type=float, default=5.0)
    ap.add_argument("--w_flex", type=float, default=20.0)
    ap.add_argument("--w_attn", type=float, default=2.0)

    # NEW: ranking loss
    ap.add_argument("--w_rank", type=float, default=20.0, help="group-wise ranking loss weight (key for pose ranking)")
    ap.add_argument("--rank_margin", type=float, default=1.0)

    # geometry params
    ap.add_argument("--contact_thresh", type=float, default=5.0)
    ap.add_argument("--interface_cutoff", type=float, default=8.0, help="used to compute pocket center/radius (native)")
    ap.add_argument("--flex_knn", type=int, default=8)
    ap.add_argument("--pocket_extra", type=float, default=2.0)

    # distance-bias params
    ap.add_argument("--num_rbf", type=int, default=16)
    ap.add_argument("--rbf_min", type=float, default=0.0)
    ap.add_argument("--rbf_max", type=float, default=30.0)
    ap.add_argument("--bias_scale", type=float, default=2.0)

    # ckpt
    ap.add_argument("--save_dir", type=str, default="./ckpts_stage3_pdbbind_pose_rank")
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained_vqmae", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/ckpts_vqmae_skempi_per_chain/ckpt_final.pt")

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = PDBbindPoseGroupDataset(
        pose_csv=args.pose_csv,
        npz_root=args.npz_root,
        K=args.K,
        seq_len=args.seq_len,
        cache_npz=True,
        interface_cutoff=args.interface_cutoff,
        pocket_margin=2.0,
        ref_xlsx=args.ref_xlsx,
        ref_sheet=args.ref_sheet,
        aff_source=args.aff_source,
        ref_aff_col=args.ref_aff_col,
        aff_clip=args.aff_clip,
        num_neg=args.num_neg,
        group_key_mode=args.group_key_mode,
        lig_sampling=args.lig_sampling,
        deterministic=args.deterministic_group_sampling,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=group_collate_fn,
    )

    model = DockingModel(
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        K=args.K,
        dropout=args.dropout,
        num_rbf=args.num_rbf,
        rbf_min=args.rbf_min,
        rbf_max=args.rbf_max,
        bias_scale=args.bias_scale,
    ).to(device)

    if args.pretrained_vqmae:
        load_pretrained_vqmae_encoders(
            model, args.pretrained_vqmae,
            d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, K=args.K,
            dropout=args.dropout, device=device
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
