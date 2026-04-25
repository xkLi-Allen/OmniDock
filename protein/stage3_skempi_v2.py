# -*- coding: utf-8 -*-
"""
Stage-3 v2: 三级对比学习（Ordinal Contrastive Learning）版本

相比 stage3_skempi.py 的核心改进：

[V2-NEG-HARD] 引入第三类"硬负样本"（CA-only poly-ALA 骨架）
  - 正样本：真实蛋白复合物（bind_label=1）
  - 软负样本（已有）：随机旋转+平移的真实蛋白链（bind_label=0）
  - 硬负样本（新增）：CA-only poly-ALA 骨架，patch feats 全零（bind_label=-1/hard）

[V2-RANK] 三级 Ordinal Ranking Loss（Margin Ranking）
  margin_soft: 正样本 logit > 软负样本 logit + margin_soft
  margin_hard: 软负样本 logit > 硬负样本 logit + margin_hard
  让 Stage-4 生成的 CA-only 骨架得到更低的评分，使 bind_logit 的排序更有意义。

[V2-POSWEIGHT] BCE 中提高正样本权重
  SKEMPI 训练集中正负样本比例约 1:2（加上硬负后 1:3），
  用 pos_weight 拉高正样本对 loss 的贡献，使模型更积极地输出高 logit。

运行示例：
  python stage3_skempi_v2.py \\
    --skempi_csv /path/to/skempi_v2.csv \\
    --npz_root /path/to/Processed_skempi_per_chain \\
    --use_negative_pose \\
    --use_hard_neg \\
    --pos_weight 4.0 \\
    --w_rank 1.5 \\
    --rank_margin_soft 3.0 \\
    --rank_margin_hard 3.0 \\
    --save_dir ./ckpts_stage3_v2
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

from unsupervised_pre_training import (  # type: ignore
    PointMLP,
    SurfFormerBlock,
    SurfVQMAE,
    set_seed,
)

# ============================================================
# 几何辅助函数（与 v1 相同）
# ============================================================

def random_rotation_matrices(batch_size: int, device: torch.device) -> torch.Tensor:
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
# 1. 数据集（与 v1 相同，完整保留）
# ============================================================

class SkempiDockingDataset(Dataset):
    GAS_CONST = 1.987e-3

    def __init__(self, skempi_csv, npz_root, K=50, seq_len=512,
                 cache_npz=True, interface_cutoff=8.0, pocket_margin=2.0):
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
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce").fillna(298.0)
        df = df[(df["Affinity_mut_parsed"] > 0) & (df["Affinity_wt_parsed"] > 0)]
        self.df = df.reset_index(drop=True)
        self._npz_cache = {}
        self.samples = []
        skipped = 0
        for row_idx, row in tqdm(self.df.iterrows(), total=len(self.df),
                                  desc='[SkempiDataset] scanning NPZ',
                                  unit='entry', dynamic_ncols=True):
            pdb_str = row["#Pdb"]
            try:
                pdb_id, rec_chains, lig_chains = self._parse_pdb_field(pdb_str)
            except ValueError:
                skipped += 1
                continue
            rec_npz = self._find_npz(pdb_id, rec_chains)
            lig_npz = self._find_npz(pdb_id, lig_chains)
            if rec_npz and lig_npz:
                self.samples.append((row_idx, rec_npz, lig_npz, pdb_str))
            else:
                skipped += 1
        print(f"[SkempiDockingDataset] usable={len(self.samples)}, skipped={skipped}")
        if not self.samples:
            raise RuntimeError("No usable samples.")

    @staticmethod
    def _parse_pdb_field(pdb_str):
        parts = str(pdb_str).split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected #Pdb format: {pdb_str}")
        return parts[0], parts[1], parts[2]

    def _find_npz(self, pdb_id, chains):
        for cand in [
            os.path.join(self.npz_root, f"{pdb_id}_{chains}.npz"),
            os.path.join(self.npz_root, f"{pdb_id}.npz"),
        ]:
            if os.path.isfile(cand):
                return cand
        return None

    def _load_surface(self, path):
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
            knn = np.concatenate([knn, np.tile(knn[:, -1:], (1, self.K - K0))], axis=1)
        elif K0 > self.K:
            knn = knn[:, :self.K]
        out = dict(xs=xs, ns=ns, centers=centers, knn=knn, order=order)
        if self.cache_npz:
            self._npz_cache[path] = out
        return out

    def _sample_window(self, xs, ns, centers, knn, order):
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
        return feats, ctrs

    def _compute_pocket_from_interface(self, rec_centers_all, lig_centers_all):
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

    def _compute_ddG(self, kd_mut, kd_wt, T):
        kd_mut, kd_wt, T = float(kd_mut), float(kd_wt), float(T)
        if kd_mut <= 0 or kd_wt <= 0:
            return 0.0
        return self.GAS_CONST * T * (math.log(kd_mut) - math.log(kd_wt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, rec_npz, lig_npz, pdb_str = self.samples[idx]
        row = self.df.iloc[row_idx]
        rec = self._load_surface(rec_npz)
        lig = self._load_surface(lig_npz)
        rec_feats, rec_centers = self._sample_window(**rec)
        lig_feats, lig_centers = self._sample_window(**lig)
        pocket_center, pocket_radius = self._compute_pocket_from_interface(
            rec["centers"], lig["centers"]
        )
        ddG = self._compute_ddG(
            row["Affinity_mut_parsed"], row["Affinity_wt_parsed"], row["Temperature"]
        )
        return {
            "rec_feats":     torch.from_numpy(rec_feats),
            "rec_centers":   torch.from_numpy(rec_centers),
            "lig_feats":     torch.from_numpy(lig_feats),
            "lig_centers":   torch.from_numpy(lig_centers),
            "pocket_center": torch.from_numpy(pocket_center),
            "pocket_radius": torch.from_numpy(pocket_radius),
            "bind_label":    torch.tensor([1.0], dtype=torch.float32),
            "affinity":      torch.tensor([ddG], dtype=torch.float32),
            "name":          pdb_str,
        }


def docking_collate_fn(batch):
    B = len(batch)
    K = batch[0]["rec_feats"].shape[1]
    Tr_max = max(b["rec_feats"].shape[0] for b in batch)
    Tl_max = max(b["lig_feats"].shape[0] for b in batch)

    def pad(feats, centers, T_max):
        T = feats.shape[0]
        if T < T_max:
            pad_f = torch.zeros((T_max - T, K, 6), dtype=feats.dtype)
            pad_c = torch.zeros((T_max - T, 3), dtype=centers.dtype)
            feats = torch.cat([feats, pad_f], dim=0)
            centers = torch.cat([centers, pad_c], dim=0)
            mask = torch.cat([torch.zeros(T, dtype=torch.bool),
                              torch.ones(T_max - T, dtype=torch.bool)], dim=0)
        else:
            mask = torch.zeros(T_max, dtype=torch.bool)
        return feats, centers, mask

    rf_list, rc_list, rm_list = [], [], []
    lf_list, lc_list, lm_list = [], [], []
    pc_list, pr_list, bl_list, af_list = [], [], [], []

    for b in batch:
        rf, rc, rm = pad(b["rec_feats"], b["rec_centers"], Tr_max)
        lf, lc, lm = pad(b["lig_feats"], b["lig_centers"], Tl_max)
        rf_list.append(rf); rc_list.append(rc); rm_list.append(rm)
        lf_list.append(lf); lc_list.append(lc); lm_list.append(lm)
        pc_list.append(b["pocket_center"])
        pr_list.append(b["pocket_radius"])
        bl_list.append(b["bind_label"])
        af_list.append(b["affinity"])

    return (
        torch.stack(rf_list), torch.stack(rc_list), torch.stack(rm_list),
        torch.stack(lf_list), torch.stack(lc_list), torch.stack(lm_list),
        torch.stack(pc_list), torch.stack(pr_list),
        torch.stack(bl_list).view(B), torch.stack(af_list).view(B),
    )


# ============================================================
# 2. 模型（与 v1 完全相同）
# ============================================================



class SurfaceEncoder(nn.Module):
    def __init__(self, in_dim=6, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.local = PointMLP(in_dim=in_dim, hidden=d_model, out_dim=d_model)
        self.blocks = nn.ModuleList([
            SurfFormerBlock(d_model=d_model, nhead=nhead, dim_ff=4*d_model, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, centers):
        x = self.local(feats)
        for blk in self.blocks:
            x = blk(x, centers, None)
        return self.norm(x)


class PocketHead(nn.Module):
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden),
            nn.GELU(), nn.Linear(hidden, 4),
        )

    def forward(self, rec_tokens, rec_mask):
        valid = ~rec_mask
        pooled = (rec_tokens * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        out = self.mlp(pooled)
        return out[:, :3], F.softplus(out[:, 3:4]) + 1e-6


class PairEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.rec_encoder = SurfaceEncoder(6, d_model, nhead, nlayers, K, dropout)
        self.lig_encoder = SurfaceEncoder(6, d_model, nhead, nlayers, K, dropout)
        self.cross_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_norm  = nn.LayerNorm(d_model)

    def forward(self, rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask):
        rec_tok = self.rec_encoder(rec_feats, rec_centers)
        lig_tok = self.lig_encoder(lig_feats, lig_centers)
        cross, attn = self.cross_attn(rec_tok, lig_tok, lig_tok, key_padding_mask=lig_mask)
        cross = self.cross_norm(cross)
        return {"rec_tokens": rec_tok, "lig_tokens": lig_tok,
                "pair_repr": cross.mean(dim=1), "cross_attn": attn}


class MultiTaskHead(nn.Module):
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        in_dim = d_model + 8
        self.bind_mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.aff_mlp  = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, pair_repr, pocket_center, pocket_radius, lig_centers, lig_mask):
        valid = ~lig_mask
        cnt = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        centroid = (lig_centers * valid.unsqueeze(-1).float()).sum(dim=1) / cnt
        diff = centroid - pocket_center
        dist = diff.norm(dim=-1, keepdim=True)
        h = torch.cat([pair_repr, pocket_center, pocket_radius, dist, diff / (dist + 1e-6)], dim=-1)
        return self.bind_mlp(h).squeeze(-1), self.aff_mlp(h).squeeze(-1)


class DockingModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=6, K=50, dropout=0.1):
        super().__init__()
        self.encoder     = PairEncoder(d_model, nhead, nlayers, K, dropout)
        self.pocket_head = PocketHead(d_model, d_model)
        self.head        = MultiTaskHead(d_model, d_model)

    def forward(self, rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask):
        enc = self.encoder(rec_feats, rec_centers, rec_mask, lig_feats, lig_centers, lig_mask)
        pc_pred, pr_pred = self.pocket_head(enc["rec_tokens"], rec_mask)
        bind_logit, aff  = self.head(enc["pair_repr"], pc_pred, pr_pred, lig_centers, lig_mask)
        return {"pocket_center_pred": pc_pred, "pocket_radius_pred": pr_pred,
                "bind_logit": bind_logit, "affinity_pred": aff,
                "rec_tokens": enc["rec_tokens"], "lig_tokens": enc["lig_tokens"],
                "rec_centers": rec_centers, "lig_centers": lig_centers,
                "rec_mask": rec_mask, "lig_mask": lig_mask}


# ============================================================
# 3. 几何约束损失
# ============================================================

def surface_complementarity_loss_pocket(
    rec_tokens, lig_tokens, rec_centers, lig_centers,
    rec_mask, lig_mask, pc_pred, pr_pred,
    contact_thresh=5.0, pocket_extra=2.0,
):
    import math as _math
    D = rec_tokens.shape[-1]
    total, used = 0.0, 0
    cd = pc_pred.detach()
    rd = pr_pred.detach()
    for b in range(rec_tokens.shape[0]):
        vr = ~rec_mask[b]
        vl = ~lig_mask[b]
        rc = rec_centers[b][vr]
        lc = lig_centers[b][vl]
        if rc.size(0) == 0 or lc.size(0) == 0:
            continue
        in_p = torch.norm(rc - cd[b], dim=-1) <= (rd[b, 0] + pocket_extra)
        if in_p.sum() == 0:
            in_p = torch.ones_like(in_p)
        rt = rec_tokens[b][vr][in_p]
        rcp = rc[in_p]
        lt = lig_tokens[b][vl]
        if rt.size(0) == 0:
            continue
        with torch.amp.autocast('cuda', enabled=False):
            Dm = torch.cdist(rcp.float(), lc.float())
        contact = (Dm <= contact_thresh).float()
        scores = (rt @ lt.t()) / _math.sqrt(D)
        pf = contact.mean().clamp(1e-4, 1.0)
        total += F.binary_cross_entropy_with_logits(scores, contact, pos_weight=(1 - pf) / pf)
        used += 1
    return rec_tokens.new_tensor(0.0) if used == 0 else total / used


# ============================================================
# 4. 三级 Ordinal Ranking Loss
# ============================================================

def ordinal_ranking_loss(logit_pos, logit_soft, logit_hard,
                         margin_soft=3.0, margin_hard=3.0):
    """
    L1 = relu(margin_soft - (pos - soft)).mean()  =>  正样本 >> 软负样本
    L2 = relu(margin_hard - (soft - hard)).mean() =>  软负样本 >> 硬负样本
    期望训练后：正样本 ~+4, 软负 ~-1, 硬负(CA-only) ~-6
    """
    l1 = F.relu(margin_soft - (logit_pos - logit_soft)).mean()
    l2 = F.relu(margin_hard - (logit_soft - logit_hard)).mean()
    return l1 + l2


# ============================================================
# 5. Loss 聚合
# ============================================================

def compute_losses(model_out, pc_gt, pr_gt, bind_label, aff_gt, is_pos, args,
                   idx_pos=None, idx_soft=None, idx_hard=None):
    bl  = model_out['bind_logit']
    pcp = model_out['pocket_center_pred']
    prp = model_out['pocket_radius_pred']
    rc  = model_out['rec_centers']
    rm  = model_out['rec_mask']

    rv = ~rm
    cnt = rv.float().sum(1, keepdim=True).clamp(1)
    rc_c = (rc * rv.unsqueeze(-1).float()).sum(1) / cnt
    l_pocket = (F.smooth_l1_loss(pcp - rc_c, pc_gt - rc_c)
                + F.smooth_l1_loss(prp, pr_gt))

    pw = torch.tensor([getattr(args, 'pos_weight', 3.0)], device=bl.device, dtype=bl.dtype)
    l_bind = F.binary_cross_entropy_with_logits(bl, bind_label, pos_weight=pw)

    l_aff = (F.smooth_l1_loss(model_out['affinity_pred'][is_pos], aff_gt[is_pos])
             if is_pos.any() else bl.new_tensor(0.0))

    l_comp = surface_complementarity_loss_pocket(
        model_out['rec_tokens'], model_out['lig_tokens'],
        rc, model_out['lig_centers'], rm, model_out['lig_mask'],
        pcp, prp, args.contact_thresh, args.pocket_extra,
    )

    l_rank = bl.new_tensor(0.0)
    if all(x is not None for x in [idx_pos, idx_soft, idx_hard]):
        n = min(len(idx_pos), len(idx_soft), len(idx_hard))
        if n > 0:
            l_rank = ordinal_ranking_loss(
                bl[idx_pos[:n]], bl[idx_soft[:n]], bl[idx_hard[:n]],
                getattr(args, 'rank_margin_soft', 3.0),
                getattr(args, 'rank_margin_hard', 3.0),
            )

    total = (args.w_pocket * l_pocket + args.w_bind * l_bind +
             args.w_aff * l_aff + args.w_comp * l_comp +
             getattr(args, 'w_rank', 1.0) * l_rank)

    return total, dict(total=total.item(), pocket=l_pocket.item(),
                       bind=l_bind.item(), aff=l_aff.item(),
                       comp=l_comp.item(), rank=l_rank.item())


# ============================================================
# 6. 训练循环
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch, args, scaler=None):
    model.train()
    total_loss = 0.0
    use_hard = getattr(args, 'use_hard_neg', True)
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for it, batch in enumerate(pbar):
        (rec_f, rec_c, rec_m, lig_f, lig_c, lig_m,
         pc_gt, pr_gt, bl_pos, aff_pos) = [x.to(device, non_blocking=True) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        B = rec_f.size(0)
        if getattr(args, 'use_negative_pose', True):
            R = random_rotation_matrices(B, device)
            com = lig_c.mean(dim=1, keepdim=True)
            centers_rot = torch.einsum('bij,btj->bti', R, lig_c - com) + com
            dv = torch.randn(B, 3, device=device)
            dv = dv / dv.norm(dim=-1, keepdim=True).clamp(1e-6)
            mag = torch.empty(B, 1, 1, device=device).uniform_(args.neg_shift_min, args.neg_shift_max)
            lig_c_soft = centers_rot + dv.unsqueeze(1) * mag
            rel = lig_f[..., :3]; nrm = lig_f[..., 3:]
            lig_f_soft = torch.cat([
                torch.einsum('bij,btkj->btki', R, rel),
                torch.einsum('bij,btkj->btki', R, nrm),
            ], dim=-1)
            if use_hard:
                # [V2-HARD-NEG] 随机混合两种硬负样本构造方法，分布更接近真实CA骨架输入
                # 方法一（CA近似feats）：用真实坐标+差分法向量近似，保持分布一致性
                # 方法三（序列打乱）：坐标/特征打乱顺序，结构失去物理意义但特征值域不变
                use_method1 = torch.rand(1).item() < 0.5  # 50% 概率选择方法一或方法三
                if use_method1:
                    # 方法一：CA-only 近似 patch feats
                    # lig_c: [B, T, 3]，用相邻点差分近似法向量
                    T = lig_c.shape[1]
                    # 中心差分近似切线方向（相邻patch center之差）
                    pad_c = torch.cat([lig_c[:, :1, :], lig_c, lig_c[:, -1:, :]], dim=1)  # [B, T+2, 3]
                    tangents = pad_c[:, 2:, :] - pad_c[:, :-2, :]                          # [B, T, 3]
                    norm_len = tangents.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    normals_approx = tangents / norm_len                                   # [B, T, 3] 单位切线
                    # rel_xyz：patch内K个点统一用patch center自身的局部偏移近似（从原始特征取）
                    # 直接复用原始 rel_xyz（lig_f[..., :3]），仅替换法向量为差分近似
                    rel_orig = lig_f[..., :3]                                              # [B, T, K, 3]
                    nrm_approx = normals_approx.unsqueeze(2).expand_as(rel_orig)           # [B, T, K, 3]
                    lig_f_hard = torch.cat([rel_orig, nrm_approx], dim=-1)                 # [B, T, K, 6]
                    lig_c_hard = lig_c.clone()                                             # 坐标不变
                else:
                    # 方法三：序列打乱（Shuffle）
                    # 特征值域与正样本完全相同，但patch顺序打乱使全局结构失去物理意义
                    T = lig_c.shape[1]
                    perm = torch.randperm(T, device=device)
                    lig_f_hard = lig_f[:, perm, :, :]   # [B, T, K, 6] 打乱token顺序
                    lig_c_hard = lig_c[:, perm, :]       # [B, T, 3] 对应坐标也打乱
            n_types = 3 if use_hard else 2
            rec_f_all = rec_f.repeat(n_types, 1, 1, 1)
            rec_c_all = rec_c.repeat(n_types, 1, 1)
            rec_m_all = rec_m.repeat(n_types, 1)
            if use_hard:
                lig_f_all = torch.cat([lig_f, lig_f_soft, lig_f_hard], dim=0)
                lig_c_all = torch.cat([lig_c, lig_c_soft, lig_c_hard], dim=0)
            else:
                lig_f_all = torch.cat([lig_f, lig_f_soft], dim=0)
                lig_c_all = torch.cat([lig_c, lig_c_soft], dim=0)
            lig_m_all = lig_m.repeat(n_types, 1)
            pc_all = pc_gt.repeat(n_types, 1)
            pr_all = pr_gt.repeat(n_types, 1)
            bl_all  = torch.cat([bl_pos] + [torch.zeros_like(bl_pos)] * (n_types - 1))
            aff_all = torch.cat([aff_pos] + [torch.zeros_like(aff_pos)] * (n_types - 1))
            is_pos_mask = torch.cat([
                torch.ones(B, dtype=torch.bool, device=device),
                torch.zeros(B * (n_types - 1), dtype=torch.bool, device=device),
            ])
            idx_pos  = list(range(0, B))
            idx_soft = list(range(B, 2 * B))
            idx_hard = list(range(2 * B, 3 * B)) if use_hard else None
        else:
            rec_f_all = rec_f; rec_c_all = rec_c; rec_m_all = rec_m
            lig_f_all = lig_f; lig_c_all = lig_c; lig_m_all = lig_m
            pc_all = pc_gt; pr_all = pr_gt; bl_all = bl_pos; aff_all = aff_pos
            is_pos_mask = torch.ones(B, dtype=torch.bool, device=device)
            idx_pos = idx_soft = idx_hard = None
        ctx = torch.amp.autocast('cuda', enabled=(scaler is not None))
        with ctx:
            out = model(rec_f_all, rec_c_all, rec_m_all, lig_f_all, lig_c_all, lig_m_all)
            loss, loss_dict = compute_losses(
                out, pc_all, pr_all, bl_all, aff_all, is_pos_mask, args,
                idx_pos=idx_pos, idx_soft=idx_soft, idx_hard=idx_hard,
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
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{total_loss / (it + 1):.4f}',
            'bind': f'{loss_dict["bind"]:.3f}',
            'rank': f'{loss_dict["rank"]:.3f}',
            'comp': f'{loss_dict["comp"]:.3f}',
        })
    return total_loss / max(1, len(loader))


# ============================================================
# 7. 加载预训练权重
# ============================================================

def load_pretrained_vqmae_encoders(model, ckpt_path, d_model, nhead, nlayers, K, dropout, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'pretrained_vqmae not found: {ckpt_path}')
    print(f'[Stage3-v2] Loading SurfVQMAE encoder from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    vq = SurfVQMAE(in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers,
                   K=K, num_codes=2048, code_dim=d_model, dropout=dropout).to(device)
    vq.load_state_dict(ckpt['model'], strict=False)
    def copy_enc(src, dst):
        dst.local.load_state_dict(src.local.state_dict())
        dst.blocks.load_state_dict(src.blocks.state_dict())
    copy_enc(vq, model.encoder.rec_encoder)
    copy_enc(vq, model.encoder.lig_encoder)
    print('[Stage3-v2] Encoder weights copied.')


# ============================================================
# 8. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='Stage3-v2: Ordinal Contrastive Learning')
    ap.add_argument('--skempi_csv',  type=str, required=True)
    ap.add_argument('--npz_root',    type=str, required=True)
    ap.add_argument('--epochs',      type=int,   default=50)
    ap.add_argument('--batch_size',  type=int,   default=2)
    ap.add_argument('--seq_len',     type=int,   default=512)
    ap.add_argument('--K',           type=int,   default=50)
    ap.add_argument('--d_model',     type=int,   default=256)
    ap.add_argument('--nhead',       type=int,   default=8)
    ap.add_argument('--nlayers',     type=int,   default=6)
    ap.add_argument('--dropout',     type=float, default=0.1)
    ap.add_argument('--lr',          type=float, default=1e-4)
    ap.add_argument('--weight_decay',type=float, default=1e-2)
    ap.add_argument('--grad_clip',   type=float, default=1.0)
    ap.add_argument('--workers',     type=int,   default=4)
    ap.add_argument('--w_pocket',    type=float, default=0.005)
    ap.add_argument('--w_bind',      type=float, default=20.0)
    ap.add_argument('--w_aff',       type=float, default=10.0)
    ap.add_argument('--w_comp',      type=float, default=5.0)
    ap.add_argument('--w_rank',      type=float, default=1.5)
    ap.add_argument('--pos_weight',  type=float, default=3.0)
    ap.add_argument('--rank_margin_soft', type=float, default=3.0)
    ap.add_argument('--rank_margin_hard', type=float, default=3.0)
    ap.add_argument('--contact_thresh',   type=float, default=5.0)
    ap.add_argument('--pocket_extra',     type=float, default=2.0)
    ap.add_argument('--use_negative_pose', action='store_true', default=True)
    ap.add_argument('--use_hard_neg',      action='store_true', default=True)
    ap.add_argument('--neg_shift_min', type=float, default=20.0)
    ap.add_argument('--neg_shift_max', type=float, default=40.0)
    ap.add_argument('--save_dir',    type=str, default='./ckpts_stage3_v2')
    ap.add_argument('--save_every',  type=int, default=1)
    ap.add_argument('--seed',        type=int, default=2023)
    ap.add_argument('--device',      type=str, default='cuda:0')
    ap.add_argument('--amp',         action='store_true')
    ap.add_argument('--pretrained_vqmae', type=str, default='')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = SkempiDockingDataset(
        skempi_csv=args.skempi_csv, npz_root=args.npz_root,
        K=args.K, seq_len=args.seq_len, cache_npz=True,
        interface_cutoff=args.contact_thresh, pocket_margin=2.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=True, collate_fn=docking_collate_fn)

    model = DockingModel(
        d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers,
        K=args.K, dropout=args.dropout,
    ).to(device)

    if args.pretrained_vqmae:
        load_pretrained_vqmae_encoders(
            model, args.pretrained_vqmae,
            d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers,
            K=args.K, dropout=args.dropout, device=device,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    print(f'[Stage3-v2] Training | pos_weight={args.pos_weight} w_rank={args.w_rank}')
    print(f'  margin_soft={args.rank_margin_soft} margin_hard={args.rank_margin_hard}')
    print(f'  use_hard_neg={args.use_hard_neg} save_dir={args.save_dir}')

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch, args, scaler=scaler)
        print(f'[Epoch {epoch}] avg_loss={avg_loss:.4f}')
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'e{epoch:03d}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optim': optimizer.state_dict(), 'args': vars(args)}, ckpt_path)
            print(f'[Stage3-v2] Saved: {ckpt_path}')

    final = os.path.join(args.save_dir, 'final.pt')
    torch.save({'epoch': args.epochs - 1, 'model': model.state_dict(),
                'optim': optimizer.state_dict(), 'args': vars(args)}, final)
    print(f'[Stage3-v2] Done. Final: {final}')


if __name__ == '__main__':
    main()
