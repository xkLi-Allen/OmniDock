# -*- coding: utf-8 -*-
"""
Stage-3: Pocket-conditioned Docking Foundation Model on SKEMPI (per-chain surfaces)
带刚体旋转 + 平移 negative pose 的微调版本

整体流程：
  1) 使用 SurfVQMAE 的 encoder 对受体表面编码，先预测 pocket center & radius
  2) 构造 batch：
        - 正样本：原始 receptor + 原始 ligand (真实 pose)
        - 负样本：同一个 receptor + 刚体随机旋转 + 平移后的 ligand (negative pose)
  3) 基于正/负样本同时训练：
        - Binding / Non-binding 分类 (正样本 label=1, 负样本 label=0)
        - Affinity / ΔΔG 回归 (只在正样本上监督)
        - Pocket 区域表面互补性约束 (正样本鼓励高互补性，负样本鼓励低互补性)
        - 局部柔性约束 (receptor / ligand token 的平滑先验)
"""

import os
import math
import argparse

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

# -----------------------------
# 复用 Stage-2 中的模块
# -----------------------------
from unsupervised_pre_training import (  # type: ignore
    PointMLP,
    SurfFormerBlock,
    SurfVQMAE,
    set_seed,
)

# ============================================================
# 一些几何辅助函数
# ============================================================

def random_rotation_matrices(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    在 SO(3) 上均匀采样 batch_size 个旋转矩阵，形状 (B,3,3)

    使用 axis-angle + Rodrigues 公式：
      - axis ~ N(0, I) 归一化
      - theta ~ Uniform(0, 2π)
      - R = I cosθ + sinθ [k]_x + (1-cosθ) k k^T
    """
    axis = torch.randn(batch_size, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B,3)

    theta = 2 * math.pi * torch.rand(batch_size, 1, device=device)  # (B,1)
    ct = torch.cos(theta).squeeze(1)  # (B,)
    st = torch.sin(theta).squeeze(1)
    vt = 1.0 - ct

    kx = axis[:, 0]
    ky = axis[:, 1]
    kz = axis[:, 2]

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
# 1. 数据集：SkempiDockingDataset（per-chain + pocket label）
# ============================================================

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SkempiDockingDataset(Dataset):
    """
    SKEMPI v2 监督数据集：
      - #Pdb : 解析出 (pdb_id, rec_chains, lig_chains)
      - npz_root 下有 {pdb_id}_{chains}.npz
      - 自动计算 pocket_center / pocket_radius
      - 计算 ΔΔG 作为 affinity 目标
    """

    GAS_CONST = 1.987e-3  # kcal / (mol*K)

    def __init__(self,
                 skempi_csv: str,
                 npz_root: str,
                 K: int = 50,
                 seq_len: int = 512,
                 cache_npz: bool = True,
                 interface_cutoff: float = 8.0,
                 pocket_margin: float = 2.0):
        super().__init__()
        assert os.path.isfile(skempi_csv), f"SKEMPI csv not found: {skempi_csv}"
        assert os.path.isdir(npz_root), f"npz_root not found: {npz_root}"

        self.npz_root = npz_root
        self.K = K
        self.seq_len = seq_len
        self.cache_npz = cache_npz
        self.interface_cutoff = interface_cutoff
        self.pocket_margin = pocket_margin

        df = pd.read_csv(skempi_csv, sep=";")
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
        df["Temperature"] = df["Temperature"].fillna(298.0)
        df = df[(df["Affinity_mut_parsed"] > 0) & (df["Affinity_wt_parsed"] > 0)]
        self.df = df.reset_index(drop=True)

        self._npz_cache = {}

        # 预先筛选所有有可用 npz 的样本
        self.samples = []  # (row_idx, rec_npz_path, lig_npz_path, pdb_str)
        total = len(self.df)
        skipped = 0

        for row_idx, row in self.df.iterrows():
            pdb_str = row["#Pdb"]
            try:
                pdb_id, rec_chains, lig_chains = self._parse_pdb_field(pdb_str)
            except ValueError:
                skipped += 1
                continue

            rec_npz = self._find_npz(pdb_id, rec_chains)
            lig_npz = self._find_npz(pdb_id, lig_chains)
            if rec_npz is not None and lig_npz is not None:
                self.samples.append((row_idx, rec_npz, lig_npz, pdb_str))
            else:
                skipped += 1

        print(f"[SkempiDockingDataset] CSV rows = {total}, usable = {len(self.samples)}, skipped_missing_npz = {skipped}")
        if len(self.samples) == 0:
            raise RuntimeError("No usable samples found; check npz_root & naming.")

    # ---------- 辅助函数 ----------

    @staticmethod
    def _parse_pdb_field(pdb_str: str):
        """
        '#Pdb' 形如 '1A4Y_A_B', '3HFM_HL_Y', '1OGA_ABC_DE'
        """
        parts = str(pdb_str).split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected #Pdb format: {pdb_str}")
        pdb_id = parts[0]
        rec_chains = parts[1]
        lig_chains = parts[2]
        return pdb_id, rec_chains, lig_chains

    def _find_npz(self, pdb_id: str, chains: str):
        """
        优先尝试 {pdb_id}_{chains}.npz
        次选 {pdb_id}.npz (如果你有老版本的整体 npz)
        """
        cand1 = os.path.join(self.npz_root, f"{pdb_id}_{chains}.npz")
        if os.path.isfile(cand1):
            return cand1
        cand2 = os.path.join(self.npz_root, f"{pdb_id}.npz")
        if os.path.isfile(cand2):
            return cand2
        return None

    def _load_surface(self, path: str):
        """
        加载 Stage1 .npz，裁剪/补齐 KNN 到固定 K。
        输出:
          xs: (M,3), ns:(M,3),
          centers:(Nc,3), knn:(Nc,K), order:(Nc,)
        """
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
                raise KeyError(f"{path} has no 'patch_order' or 'patch_morton'")

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

    def _sample_window(self, xs, ns, centers, knn, order):
        """
        在 Morton 顺序上截取一段窗口，拼成 (T,K,6)/(T,3)。
        """
        Nc = centers.shape[0]
        if Nc <= self.seq_len:
            sel = order
        else:
            start = np.random.randint(0, Nc - self.seq_len + 1)
            sel = order[start:start + self.seq_len]

        pts_idx = knn[sel]         # (T,K)
        ctrs = centers[sel]        # (T,3)
        rel_xyz = xs[pts_idx] - ctrs[:, None, :]
        norms = ns[pts_idx]
        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)  # (T,K,6)
        return feats, ctrs

    def _compute_pocket_from_interface(self, rec_centers_all, lig_centers_all):
        """
        几何口袋估计（用于 supervision 的“真口袋”）
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

    def _generate_negative_sample(self, rec_feats, lig_feats, rec_centers, lig_centers):
        """
        改进后的负样本生成策略：
        - 大角度随机旋转（全 SO(3) 均匀采样）确保彻底改变朝向
        - 大幅平移（20-40 Å）使配体远离受体结合口袋
        - 加入随机内部扰动（改变 patch 间的相对坐标），
          使模型无法仅凭整体位置判断正负样本
        """
        device = rec_feats.device
        B = rec_feats.size(0)
        rec_centers = rec_centers.detach()

        # 1. 全 SO(3) 随机旋转（不限小角度，彻底打乱朝向）
        R = random_rotation_matrices(B, device)  # (B,3,3)

        # 计算配体中心
        lig_centroid = lig_centers.mean(dim=1, keepdim=True)  # (B,1,3)
        # 绕配体质心旋转
        centered = lig_centers - lig_centroid  # (B,T,3)
        rotated = torch.einsum('bij,btj->bti', R, centered)  # (B,T,3)
        new_lig_centers = rotated + lig_centroid

        # 2. 大幅平移（确保远离口袋）
        # 使用受体中心 + 随机方向的大位移
        rec_centroid = rec_centers.mean(dim=1)  # (B,3)
        direction = torch.randn(B, 3, device=device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        shift_dist = 20.0 + 20.0 * torch.rand(B, 1, device=device)  # 20-40 Å
        shift = direction * shift_dist  # (B,3)
        new_lig_centers = new_lig_centers + shift.unsqueeze(1)  # (B,T,3)

        # 3. 不加内部扰动 - 保持刚体变换，让模型通过几何互补性学习binding
        # 原来的 internal_noise 会破坏 patch 间相对几何，导致 flex loss 对正负样本都很高
        # internal_noise = torch.randn_like(new_lig_centers) * 3.0  # REMOVED
        # new_lig_centers = new_lig_centers + internal_noise  # REMOVED

        return new_lig_centers

    def _sample_other_ligands(self, rec_chains, lig_chains):
        """
        采样不与当前受体配对的其他配体
        """
        # 获取所有其他配体
        other_samples = self.df[(self.df["#Pdb"] != f"{rec_chains}_{lig_chains}")]

        # 随机选择不与当前受体配对的配体
        random_idx = torch.randint(0, len(other_samples), (1,)).item()
        other_sample = other_samples.iloc[random_idx]
        other_pdb_str = other_sample["#Pdb"]

        # 解析出与当前受体不配对的配体
        other_pdb_id, other_rec_chains, other_lig_chains = self._parse_pdb_field(other_pdb_str)
        lig_npz_path = self._find_npz(other_pdb_id, other_lig_chains)

        return lig_npz_path
        
    def _compute_ddG(self, kd_mut, kd_wt, T):
        """
        ΔΔG = ΔG_mut - ΔG_wt, ΔG = R T ln Kd
        """
        kd_mut = float(kd_mut)
        kd_wt = float(kd_wt)
        T = float(T)
        if kd_mut <= 0 or kd_wt <= 0:
            return 0.0
        dG_mut = self.GAS_CONST * T * math.log(kd_mut)
        dG_wt = self.GAS_CONST * T * math.log(kd_wt)
        return dG_mut - dG_wt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, rec_npz, lig_npz, pdb_str = self.samples[idx]
        row = self.df.iloc[row_idx]

        rec = self._load_surface(rec_npz)
        lig = self._load_surface(lig_npz)

        rec_feats, rec_centers = self._sample_window(**rec)
        lig_feats, lig_centers = self._sample_window(**lig)

        # 用 “全局 centers” 算真口袋（监督信号）
        pocket_center, pocket_radius = self._compute_pocket_from_interface(
            rec["centers"], lig["centers"]
        )

        # Skempi 原始样本全部视为 binder
        bind_label = np.array([1.0], dtype=np.float32)
        ddG = self._compute_ddG(
            row["Affinity_mut_parsed"], row["Affinity_wt_parsed"], row["Temperature"]
        )
        affinity = np.array([ddG], dtype=np.float32)

        # 获取与当前受体配对不上的其他配体（其他配体）
        other_ligand_path = self._sample_other_ligands(row["#Pdb"].split("_")[1], row["#Pdb"].split("_")[2])

        return {
            "rec_feats": torch.from_numpy(rec_feats),
            "rec_centers": torch.from_numpy(rec_centers),
            "lig_feats": torch.from_numpy(lig_feats),
            "lig_centers": torch.from_numpy(lig_centers),
            "pocket_center": torch.from_numpy(pocket_center),
            "pocket_radius": torch.from_numpy(pocket_radius),
            "bind_label": torch.from_numpy(bind_label),
            "affinity": torch.from_numpy(affinity),
            "other_ligand": other_ligand_path,
            "name": pdb_str,
        }



def docking_collate_fn(batch):
    """
    pad Tr / Tl 到 batch 内最大长度
    """
    B = len(batch)
    K = batch[0]["rec_feats"].shape[1]
    Tr_max = max(b["rec_feats"].shape[0] for b in batch)
    Tl_max = max(b["lig_feats"].shape[0] for b in batch)

    rec_feats_list, rec_centers_list, rec_mask_list = [], [], []
    lig_feats_list, lig_centers_list, lig_mask_list = [], [], []
    pocket_centers, pocket_radii, bind_labels, affinities = [], [], [], []

    for b in batch:
        # receptor
        rec_feats = b["rec_feats"]
        rec_centers = b["rec_centers"]
        Tr = rec_feats.shape[0]
        if Tr < Tr_max:
            pad_r = Tr_max - Tr
            pad_feats = torch.zeros((pad_r, K, 6), dtype=rec_feats.dtype)
            pad_centers = torch.zeros((pad_r, 3), dtype=rec_centers.dtype)
            rec_feats = torch.cat([rec_feats, pad_feats], dim=0)
            rec_centers = torch.cat([rec_centers, pad_centers], dim=0)
            mask_r = torch.cat([
                torch.zeros(Tr, dtype=torch.bool),
                torch.ones(pad_r, dtype=torch.bool),
            ], dim=0)
        else:
            mask_r = torch.zeros(Tr_max, dtype=torch.bool)
        rec_feats_list.append(rec_feats)
        rec_centers_list.append(rec_centers)
        rec_mask_list.append(mask_r)

        # ligand
        lig_feats = b["lig_feats"]
        lig_centers = b["lig_centers"]
        Tl = lig_feats.shape[0]
        if Tl < Tl_max:
            pad_l = Tl_max - Tl
            pad_feats = torch.zeros((pad_l, K, 6), dtype=lig_feats.dtype)
            pad_centers = torch.zeros((pad_l, 3), dtype=lig_centers.dtype)
            lig_feats = torch.cat([lig_feats, pad_feats], dim=0)
            lig_centers = torch.cat([lig_centers, pad_centers], dim=0)
            mask_l = torch.cat([
                torch.zeros(Tl, dtype=torch.bool),
                torch.ones(pad_l, dtype=torch.bool),
            ], dim=0)
        else:
            mask_l = torch.zeros(Tl_max, dtype=torch.bool)
        lig_feats_list.append(lig_feats)
        lig_centers_list.append(lig_centers)
        lig_mask_list.append(mask_l)

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
# 2. 模型：先预测口袋，再做 pocket-conditioned docking
# ============================================================

class SurfaceEncoder(nn.Module):
    """
    只保留 SurfFormer encoder 部分（PointMLP + 多层 SurfFormerBlock）
    """
    def __init__(self, in_dim=6, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.local = PointMLP(in_dim=in_dim, hidden=d_model, out_dim=d_model)
        self.blocks = nn.ModuleList([
            SurfFormerBlock(d_model=d_model, nhead=nhead, dim_ff=4 * d_model, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, centers):
        """
        feats: (B,T,K,6)
        centers: (B,T,3)
        """
        x = self.local(feats)   # (B,T,D)
        key_padding = None
        for blk in self.blocks:
            x = blk(x, centers, key_padding)
        x = self.norm(x)
        return x


class PocketHead(nn.Module):
    """
    只用 receptor tokens 先预测 pocket center & radius
    """
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, rec_tokens, rec_mask):
        """
        rec_tokens: (B,Tr,D)
        rec_mask: (B,Tr) True=pad
        """
        B, Tr, D = rec_tokens.shape
        valid = ~rec_mask  # (B,Tr)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (rec_tokens * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

        out = self.mlp(pooled)  # (B,4)
        center = out[:, :3]
        log_r = out[:, 3:4]
        radius = F.softplus(log_r) + 1e-6
        return center, radius


class PairEncoder(nn.Module):
    """
    受体/配体表面编码 + cross-attention 得到 pair 表示
    """
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.rec_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead,
                                          nlayers=nlayers, K=K, dropout=dropout)
        self.lig_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead,
                                          nlayers=nlayers, K=K, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

    def forward(self,
                rec_feats, rec_centers, rec_mask,
                lig_feats, lig_centers, lig_mask):
        rec_tokens = self.rec_encoder(rec_feats, rec_centers)  # (B,Tr,D)
        lig_tokens = self.lig_encoder(lig_feats, lig_centers)  # (B,Tl,D)

        cross, attn = self.cross_attn(
            query=rec_tokens,
            key=lig_tokens,
            value=lig_tokens,
            key_padding_mask=lig_mask,
        )  # (B,Tr,D)

        cross = self.cross_norm(cross)
        pair_repr = cross.mean(dim=1)  # (B,D)

        return {
            "rec_tokens": rec_tokens,
            "lig_tokens": lig_tokens,
            "cross_tokens": cross,
            "pair_repr": pair_repr,
            "cross_attn": attn,
        }


class MultiTaskHead(nn.Module):
    """
    口袋条件 Multi-task head:
      输入: concat(pair_repr, pocket_center, pocket_radius,
                   lig_centroid_dist, lig_centroid_dir(3)) -> (D+8)
      输出:
        - binding logit
        - affinity
    """
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        in_dim = d_model + 8  # +4 pocket + 1 dist + 3 dir
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

    def forward(self, pair_repr, pocket_center, pocket_radius,
                lig_centers, lig_mask):
        # 配体质心
        valid = ~lig_mask  # (B,Tl)
        cnt = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        lig_centroid = (lig_centers * valid.unsqueeze(-1).float()).sum(dim=1) / cnt  # (B,3)

        # 配体质心 -> 预测口袋中心 的距离和方向（关键几何特征）
        diff = lig_centroid - pocket_center  # (B,3)
        dist = diff.norm(dim=-1, keepdim=True)  # (B,1)
        direction = diff / (dist + 1e-6)         # (B,3) 单位方向向量

        pocket_feat = torch.cat([pocket_center, pocket_radius], dim=-1)  # (B,4)
        geo_feat = torch.cat([dist, direction], dim=-1)                  # (B,4)
        h = torch.cat([pair_repr, pocket_feat, geo_feat], dim=-1)        # (B,D+8)
        bind_logit = self.bind_mlp(h).squeeze(-1)
        affinity = self.aff_mlp(h).squeeze(-1)
        return bind_logit, affinity


class DockingModel(nn.Module):
    """
    整体：
      - PairEncoder 得到 rec_tokens / lig_tokens / pair_repr
      - PocketHead 用 rec_tokens 预测 pocket (先做)
      - MultiTaskHead 基于 pocket-conditioned pair_repr 做 binding/affinity
    """
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.encoder = PairEncoder(d_model=d_model, nhead=nhead, nlayers=nlayers,
                                   K=K, dropout=dropout)
        self.pocket_head = PocketHead(d_model=d_model, hidden=d_model)
        self.head = MultiTaskHead(d_model=d_model, hidden=d_model)

    def forward(self,
                rec_feats, rec_centers, rec_mask,
                lig_feats, lig_centers, lig_mask):
        enc_out = self.encoder(
            rec_feats, rec_centers, rec_mask,
            lig_feats, lig_centers, lig_mask
        )
        rec_tokens = enc_out["rec_tokens"]
        pair_repr = enc_out["pair_repr"]

        pocket_center_pred, pocket_radius_pred = self.pocket_head(rec_tokens, rec_mask)
        bind_logit, affinity_pred = self.head(
            pair_repr, pocket_center_pred, pocket_radius_pred,
            lig_centers, lig_mask
        )

        out = {
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
        return out


# ============================================================
# 3. 几何约束：Pocket-conditioned surface complementarity & flex
# ============================================================

def surface_complementarity_loss_pocket(
    rec_tokens, lig_tokens, rec_centers, lig_centers,
    rec_mask, lig_mask,
    pocket_center_pred, pocket_radius_pred,
    contact_thresh=5.0,
    pocket_extra=2.0,
):
    """
    Pocket-conditioned 表面互补性约束：

      1) 用 predicted pocket center/radius 选 receptor pocket 内 patch
      2) 在这些 receptor patch 与所有 ligand patch 上做 contact-map BCE

    正样本：由于 ligand 在真实接口附近，存在较多 contact=1，鼓励高相似度
    负样本：ligand 被刚体旋转 + 大平移到远处，几乎无 contact，鼓励整体相似度→0
    """
    device = rec_tokens.device
    B, Tr, D = rec_tokens.shape
    Tl = lig_tokens.shape[1]

    total_loss = 0.0
    used = 0

    center_det = pocket_center_pred.detach()   # (B,3)
    radius_det = pocket_radius_pred.detach()   # (B,1)

    for b in range(B):
        valid_r = ~rec_mask[b]  # (Tr,)
        valid_l = ~lig_mask[b]  # (Tl,)

        r_ctrs = rec_centers[b][valid_r]   # (Tr_v,3)
        l_ctrs = lig_centers[b][valid_l]   # (Tl_v,3)
        if r_ctrs.size(0) == 0 or l_ctrs.size(0) == 0:
            continue

        pc = center_det[b]                # (3,)
        pr = radius_det[b, 0]             # scalar tensor
        dist_p = torch.norm(r_ctrs - pc.unsqueeze(0), dim=-1)  # (Tr_v,)
        in_pocket = dist_p <= (pr + pocket_extra)

        if in_pocket.sum().item() == 0:
            in_pocket = torch.ones_like(dist_p, dtype=torch.bool, device=device)

        r_tok = rec_tokens[b][valid_r][in_pocket]  # (Tr_p,D)
        r_ctrs_p = r_ctrs[in_pocket]               # (Tr_p,3)
        if r_tok.size(0) == 0:
            continue

        l_tok = lig_tokens[b][valid_l]             # (Tl_v,D)

        with torch.amp.autocast('cuda', enabled=False):
            Dmat = torch.cdist(r_ctrs_p.float(), l_ctrs.float())  # (Tr_p,Tl_v)
        contact = (Dmat <= contact_thresh).float().to(device)     # (Tr_p,Tl_v)

        scores = (r_tok @ l_tok.t()) / math.sqrt(D)               # (Tr_p,Tl_v)

        if contact.numel() == 0:
            continue

        pos_frac = contact.mean().clamp(min=1e-4, max=1.0)
        pos_weight = (1.0 - pos_frac) / pos_frac
        bce = F.binary_cross_entropy_with_logits(scores, contact,
                                                 pos_weight=pos_weight)
        total_loss += bce
        used += 1

    if used == 0:
        return rec_tokens.new_tensor(0.0)
    return total_loss / used


def local_flexibility_loss(tokens, centers, mask, k_neighbors=8):
    """
    局部结构约束：
      1. CA-CA 键长约束：相邻 patch center 间距应接近 3.8 Å（真实 CA-CA 键长）
         这比原来的 token 平滑 loss 更有蛋白质几何意义。
      2. 保留 KNN token 平滑作为辅助正则（权重降低）
    """
    device = tokens.device
    B, T, D = tokens.shape
    total = 0.0
    used = 0
    CA_CA_IDEAL = 3.8  # Å, ideal CA-CA distance along backbone

    for b in range(B):
        valid = ~mask[b]      # (T,)
        h = tokens[b][valid]  # (Tv,D)
        c = centers[b][valid] # (Tv,3)
        Tv = h.shape[0]
        if Tv <= 1:
            continue

        # --- 1. CA-CA bond length constraint on sequential patch centers ---
        # Adjacent patch centers should be ~3.8 Å apart (backbone CA-CA distance)
        if Tv >= 2:
            ca_dists = torch.norm(c[1:] - c[:-1], dim=-1)  # (Tv-1,)
            target = torch.full_like(ca_dists, CA_CA_IDEAL)
            l_bond = F.smooth_l1_loss(ca_dists, target)
            total += 2.0 * l_bond  # higher weight: direct geometric constraint

        # --- 2. KNN token smoothness (reduced weight, auxiliary regularization) ---
        with torch.amp.autocast('cuda', enabled=False):
            Dmat = torch.cdist(c.float().to(device), c.float().to(device))
        k = min(k_neighbors + 1, Tv)
        _, nn_idx = torch.topk(Dmat, k=k, dim=-1, largest=False)
        nn_idx = nn_idx[:, 1:]  # exclude self

        src = h.unsqueeze(1)      # (Tv,1,D)
        nbr = h[nn_idx]           # (Tv,k,D)
        diff = src - nbr
        l_smooth = diff.pow(2).mean()
        total += 0.3 * l_smooth  # reduced weight vs original 1.0
        used += 1

    if used == 0:
        return tokens.new_tensor(0.0)
    return total / used


def protein_structure_constraints(lig_centers, lig_mask):
    """
    蛋白质结构约束（作用于配体 patch centers，即 CA 原子位置）：
      1. 紧凑性约束（Radius of Gyration）
      2. 避免自相交（非相邻 CA 原子最小距离 >= 2.8 Å）
      3. 局部螺旋间距约束（CA_i -> CA_{i+3} ≈ 5.4 Å, CA_i -> CA_{i+4} ≈ 6.2 Å）
    
    这些约束确保模型学到的配体特征符合真实蛋白质的几何约束，
    而不仅仅是学习"如何与受体相互作用"。
    """
    device = lig_centers.device
    B, T, _ = lig_centers.shape

    total_loss = lig_centers.new_tensor(0.0)
    used = 0

    for b in range(B):
        valid = ~lig_mask[b]          # (T,)
        ca = lig_centers[b][valid]    # (Tv,3)
        Tv = ca.shape[0]

        if Tv < 2:
            continue

        # 0. CA-CA 键长约束：相邻 CA 距离应接近 3.8 Å [NEW - highest priority]
        # 这是最直接的蛋白质骨架几何约束
        ca_seq_dists = torch.norm(ca[1:] - ca[:-1], dim=-1)  # (Tv-1,)
        l_ca_bond = F.smooth_l1_loss(
            ca_seq_dists,
            torch.full_like(ca_seq_dists, 3.8)
        )
        total_loss = total_loss + 3.0 * l_ca_bond

        if Tv < 5:
            used += 1
            continue

        # 1. 紧凑性约束：Radius of Gyration
        center = ca.mean(dim=0, keepdim=True)
        rg = torch.sqrt(torch.mean(torch.sum((ca - center) ** 2, dim=1)) + 1e-8)
        # 预期 Rg ≈ 2.2 + 1.1 * ln(L) (经验公式)
        target_rg = 2.2 + 1.1 * torch.log(
            torch.tensor(float(Tv), device=device, dtype=ca.dtype)
        )
        l_compact = F.relu(rg - target_rg) ** 2
        total_loss = total_loss + 0.3 * l_compact

        # 2. 避免自相交：非相邻 CA 原子最小距离 >= 2.8 Å
        if Tv >= 4:
            ca_dmat = torch.cdist(ca, ca)
            idx = torch.arange(Tv, device=device)
            sep = (idx[:, None] - idx[None, :]).abs()
            clash_mask = (sep >= 2) & torch.triu(
                torch.ones(Tv, Tv, dtype=torch.bool, device=device), diagonal=1
            )
            if clash_mask.any():
                clash_dist = torch.tensor(2.8, device=device, dtype=ca.dtype)
                bad = F.relu(clash_dist - ca_dmat[clash_mask])
                l_clash = (bad * bad).mean()
                total_loss = total_loss + 1.5 * l_clash

        # 3. 局部螺旋间距约束
        if Tv >= 4:
            # Alpha helix: CA_i -> CA_{i+3} ≈ 5.4 Å
            d_i3 = torch.norm(ca[3:] - ca[:-3], dim=1)
            l_helix_i3 = F.mse_loss(d_i3, torch.full_like(d_i3, 5.4))
            total_loss = total_loss + 0.5 * l_helix_i3

        if Tv >= 5:
            # CA_i -> CA_{i+4} ≈ 6.2 Å
            d_i4 = torch.norm(ca[4:] - ca[:-4], dim=1)
            l_helix_i4 = F.mse_loss(d_i4, torch.full_like(d_i4, 6.2))
            total_loss = total_loss + 0.3 * l_helix_i4

        used += 1

    if used == 0:
        return lig_centers.new_tensor(0.0)
    return total_loss / used


# ============================================================
# 4. Loss 聚合 & Train loop
# ============================================================

def compute_losses(model_out,
                   pocket_center_gt, pocket_radius_gt,
                   bind_label, affinity_gt,
                   is_pos,
                   args):
    """
    聚合所有任务的 loss：
      - pocket center/radius 回归（正/负样本都监督，因为 pocket 是 receptor 属性）
      - binding BCE（正样本 label=1, 负样本 label=0）
      - affinity SmoothL1（只在正样本上监督）
      - pocket-conditioned surface complementarity（正/负几何不同，产生对比）
      - local flexibility (receptor + ligand, 正/负都加平滑)
      - protein structure constraints (NEW: 紧凑性、避免自相交、螺旋间距)
    """
    pc_pred = model_out["pocket_center_pred"]  # (B,3)
    pr_pred = model_out["pocket_radius_pred"]  # (B,1)
    bind_logit = model_out["bind_logit"]       # (B,)
    aff_pred = model_out["affinity_pred"]      # (B,)

    rec_tokens = model_out["rec_tokens"]
    lig_tokens = model_out["lig_tokens"]
    rec_centers = model_out["rec_centers"]
    lig_centers = model_out["lig_centers"]
    rec_mask = model_out["rec_mask"]
    lig_mask = model_out["lig_mask"]

    # pocket regression：正负样本都监督
    # 归一化：相对于 receptor 质心计算偏移，避免绝对坐标 MSE 过大
    rec_valid_mask = ~rec_mask  # (B, Tr), True = valid token
    rec_cnt = rec_valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
    rec_centroid = (rec_centers * rec_valid_mask.unsqueeze(-1).float()).sum(dim=1) / rec_cnt  # (B,3)
    pc_pred_rel = pc_pred - rec_centroid          # 预测偏移（相对质心）
    pc_gt_rel   = pocket_center_gt - rec_centroid # GT 偏移（相对质心）
    l_pocket_center = F.smooth_l1_loss(pc_pred_rel, pc_gt_rel)
    l_pocket_radius = F.smooth_l1_loss(pr_pred, pocket_radius_gt)
    l_pocket = l_pocket_center + l_pocket_radius

    # binding classification：正样本 label=1, 负样本 label=0
    l_bind = F.binary_cross_entropy_with_logits(bind_logit, bind_label)

    # affinity regression：只在正样本上监督
    if is_pos.any():
        l_aff = F.smooth_l1_loss(aff_pred[is_pos], affinity_gt[is_pos])
    else:
        l_aff = aff_pred.new_tensor(0.0)

    # pocket-conditioned surface complementarity：正/负都参与
    l_comp = surface_complementarity_loss_pocket(
        rec_tokens, lig_tokens,
        rec_centers, lig_centers,
        rec_mask, lig_mask,
        pc_pred, pr_pred,
        contact_thresh=args.contact_thresh,
        pocket_extra=args.pocket_extra,
    )

    # local flexibility — disabled: KNN token smoothness on random-init tokens
    # produces unlearnable noise gradients; patch centers are not CA atoms.
    l_flex = rec_tokens.new_tensor(0.0)

    # protein structure constraints — disabled: patch centers are surface sample
    # points, not CA atoms; helix/clash constraints are meaningless here.
    l_struct = lig_centers.new_tensor(0.0)

    # Weight for structure constraint (default 0.5 if not in args)
    w_struct = getattr(args, 'w_struct', 0.5)

    total = (args.w_pocket * l_pocket +
             args.w_bind * l_bind +
             args.w_aff * l_aff +
             args.w_comp * l_comp +
             args.w_flex * l_flex +
             w_struct * l_struct)

    loss_dict = dict(
        total=total.item(),
        pocket=l_pocket.item(),
        bind=l_bind.item(),
        aff=l_aff.item(),
        comp=l_comp.item(),
        flex=l_flex.item(),
        struct=l_struct.item(),
    )
    return total, loss_dict


def train_one_epoch(model, loader, optimizer, device, epoch, args, scaler=None):
    model.train()
    total = 0.0
    iters = len(loader)
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

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

        B = rec_feats.size(0)

        if args.use_negative_pose:
            # ---------- 正样本：原始 pose ----------
            rec_feats_pos = rec_feats
            rec_centers_pos = rec_centers
            rec_mask_pos = rec_mask

            lig_feats_pos = lig_feats
            lig_centers_pos = lig_centers
            lig_mask_pos = lig_mask

            # ---------- 负样本：刚体随机旋转 + 大平移后的配体 ----------
            R = random_rotation_matrices(B, device)  # (B,3,3)

            # 以 ligand patch centers 的质心为中心旋转
            com = lig_centers.mean(dim=1, keepdim=True)          # (B,1,3)
            centers_rel = lig_centers - com                      # (B,Tl,3)
            # new_centers = R * centers_rel + com
            #  centers_rel: (B,Tl,3), R: (B,3,3) => 'bij,btj->bti'
            centers_rot = torch.einsum('bij,btj->bti', R, centers_rel)  # (B,Tl,3)
            centers_rot = centers_rot + com                            # 回到全局

            # 随机方向 + 大平移距离 [neg_shift_min, neg_shift_max]
            dir_vec = torch.randn(B, 3, device=device)
            dir_vec = dir_vec / dir_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            mag = torch.empty(B, 1, 1, device=device).uniform_(
                args.neg_shift_min, args.neg_shift_max
            )
            shift = dir_vec.view(B, 1, 3) * mag                     # (B,1,3)
            lig_centers_neg = centers_rot + shift                   # (B,Tl,3)

            # 对局部点 rel_xyz 和 normals 同样做旋转
            rel = lig_feats[..., :3]   # (B,Tl,K,3)
            nrm = lig_feats[..., 3:]   # (B,Tl,K,3)

            # new_rel[b,t,k,i] = Σ_j R[b,i,j] * rel[b,t,k,j]
            new_rel = torch.einsum('bij,btkj->btki', R, rel)
            new_nrm = torch.einsum('bij,btkj->btki', R, nrm)
            lig_feats_neg = torch.cat([new_rel, new_nrm], dim=-1)  # (B,Tl,K,6)

            rec_feats_neg = rec_feats_pos
            rec_centers_neg = rec_centers_pos
            rec_mask_neg = rec_mask_pos

            lig_mask_neg = lig_mask_pos

            # 拼接正/负样本
            rec_feats_all = torch.cat([rec_feats_pos, rec_feats_neg], dim=0)
            rec_centers_all = torch.cat([rec_centers_pos, rec_centers_neg], dim=0)
            rec_mask_all = torch.cat([rec_mask_pos, rec_mask_neg], dim=0)

            lig_feats_all = torch.cat([lig_feats_pos, lig_feats_neg], dim=0)
            lig_centers_all = torch.cat([lig_centers_pos, lig_centers_neg], dim=0)
            lig_mask_all = torch.cat([lig_mask_pos, lig_mask_neg], dim=0)

            pocket_center_gt_all = torch.cat([pocket_center_gt, pocket_center_gt], dim=0)
            pocket_radius_gt_all = torch.cat([pocket_radius_gt, pocket_radius_gt], dim=0)

            # binding label：正=1，负=0
            bind_label_neg = torch.zeros_like(bind_label_pos)
            bind_label_all = torch.cat([bind_label_pos, bind_label_neg], dim=0)

            # affinity：只在正样本上监督，负样本 dummy=0
            affinity_gt_neg = torch.zeros_like(affinity_gt_pos)
            affinity_gt_all = torch.cat([affinity_gt_pos, affinity_gt_neg], dim=0)

            is_pos = torch.cat([
                torch.ones_like(bind_label_pos, dtype=torch.bool),
                torch.zeros_like(bind_label_pos, dtype=torch.bool)
            ], dim=0)
        else:
            # 不使用负样本，退化为原始正样本训练
            rec_feats_all = rec_feats
            rec_centers_all = rec_centers
            rec_mask_all = rec_mask

            lig_feats_all = lig_feats
            lig_centers_all = lig_centers
            lig_mask_all = lig_mask

            pocket_center_gt_all = pocket_center_gt
            pocket_radius_gt_all = pocket_radius_gt
            bind_label_all = bind_label_pos
            affinity_gt_all = affinity_gt_pos

            is_pos = torch.ones_like(bind_label_pos, dtype=torch.bool)

        ctx = torch.amp.autocast('cuda', enabled=(scaler is not None))
        with ctx:
            out = model(
                rec_feats_all, rec_centers_all, rec_mask_all,
                lig_feats_all, lig_centers_all, lig_mask_all,
            )
            loss, loss_dict = compute_losses(
                out,
                pocket_center_gt_all, pocket_radius_gt_all,
                bind_label_all, affinity_gt_all,
                is_pos,
                args,
            )

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

        total += loss.item()
        avg_loss = total / (it + 1)
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "pocket": f"{loss_dict['pocket']:.3f}",
            "bind": f"{loss_dict['bind']:.3f}",
            "aff": f"{loss_dict['aff']:.3f}",
            "comp": f"{loss_dict['comp']:.3f}",
            "flex": f"{loss_dict['flex']:.3f}",
        })

    return total / max(1, iters)


# ============================================================
# 5. 从 SurfVQMAE 加载 encoder 权重
# ============================================================

def load_pretrained_vqmae_encoders(model: DockingModel,
                                   ckpt_path: str,
                                   d_model: int,
                                   nhead: int,
                                   nlayers: int,
                                   K: int,
                                   dropout: float,
                                   device):
    """
    从 SurfVQMAE ckpt 中拷贝 encoder (local + blocks) 参数到
    receptor / ligand 的 SurfaceEncoder。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained_vqmae not found: {ckpt_path}")

    print(f"[Stage3] Loading SurfVQMAE encoder from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    vq = SurfVQMAE(
        in_dim=6, d_model=d_model, nhead=nhead,
        nlayers=nlayers, K=K,
        num_codes=2048, code_dim=d_model,
        dropout=dropout,
    ).to(device)
    vq.load_state_dict(ckpt["model"], strict=False)

    def copy_encoder(src: SurfVQMAE, dst: SurfaceEncoder):
        dst.local.load_state_dict(src.local.state_dict())
        dst.blocks.load_state_dict(src.blocks.state_dict())

    copy_encoder(vq, model.encoder.rec_encoder)
    copy_encoder(vq, model.encoder.lig_encoder)
    print("[Stage3] Encoder weights copied into receptor & ligand encoders.")


# ============================================================
# 6. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Stage3 pocket-conditioned docking on SKEMPI (with rigid-body negative poses)")

    ap.add_argument(
        "--skempi_csv",
        type=str,
        default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/skempi_v2.csv"
    )
    ap.add_argument(
        "--npz_root",
        type=str,
        default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/Processed_skempi_per_chain"
    )

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--K", type=int, default=50)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)

    # loss 权重
    ap.add_argument("--w_pocket", type=float, default=0.005)
    ap.add_argument("--w_bind", type=float, default=20)
    ap.add_argument("--w_aff", type=float, default=10)
    ap.add_argument("--w_comp", type=float, default=5)
    ap.add_argument("--w_flex", type=float, default=20)
    ap.add_argument("--w_struct", type=float, default=0.5,
                    help="Weight for protein structure constraints (compactness, clash, helix spacing)")

    # 几何超参
    ap.add_argument("--contact_thresh", type=float, default=5.0,
                    help="Å, receptor/ligand patch center contact threshold")
    ap.add_argument("--flex_knn", type=int, default=8)
    ap.add_argument("--pocket_extra", type=float, default=2.0,
                    help="Å, pocket radius 的额外 buffer，用于几何损失")

    # negative pose 设置（刚体平移距离范围）
    ap.add_argument("--use_negative_pose", action="store_true",
                    help="是否在训练中构造 negative pose (receptor + rotated+shifted ligand)")
    ap.add_argument("--neg_shift_min", type=float, default=20.0,
                    help="negative pose 平移距离最小值 (Å)")
    ap.add_argument("--neg_shift_max", type=float, default=40.0,
                    help="negative pose 平移距离最大值 (Å)")

    ap.add_argument("--save_dir", type=str, default="./ckpts_stage3_pocket_conditioned_negpose_rot")
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--pretrained_vqmae", type=str, default="",
                    help="SurfVQMAE ckpt 用于初始化 encoder")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = SkempiDockingDataset(
        skempi_csv=args.skempi_csv,
        npz_root=args.npz_root,
        K=args.K,
        seq_len=args.seq_len,
        cache_npz=True,
        interface_cutoff=args.contact_thresh,
        pocket_margin=2.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

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
