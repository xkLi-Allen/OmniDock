# -*- coding: utf-8 -*-
"""
Stage 3 Model: Pocket-conditioned Docking with Structure Generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_components import SurfaceEncoder, BackboneEncoder, TorsionHead, SSHead


def random_rotation_matrices(batch_size, device):
    """Sample uniform SO(3) rotation matrices."""
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


class PocketHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, rec_tokens, rec_pad):
        valid = ~rec_pad
        denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (rec_tokens * valid.unsqueeze(-1).float()).sum(dim=1) / denom
        out = self.mlp(pooled)
        center = out[:, :3]
        radius = F.softplus(out[:, 3:4]) + 1e-6
        return center, radius


class BindingHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        in_dim = d_model + 4
        self.bind_mlp = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.aff_mlp = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )

    def forward(self, pair_repr, pocket_center, pocket_radius, lig_centers, lig_pad):
        valid = ~lig_pad
        cnt = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        lig_centroid = (lig_centers * valid.unsqueeze(-1).float()).sum(dim=1) / cnt
        diff = lig_centroid - pocket_center
        dist = diff.norm(dim=-1, keepdim=True)
        geo_feat = torch.cat([pocket_center, dist], dim=-1)
        h = torch.cat([pair_repr, geo_feat], dim=-1)
        return self.bind_mlp(h).squeeze(-1), self.aff_mlp(h).squeeze(-1)


class TorsionDenoisingHead(nn.Module):
    """Pocket-conditioned torsion denoising."""
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = TorsionHead(d_model=d_model, hidden=d_model)

    def forward(self, lig_bb_tokens, rec_tokens, rec_pad):
        cross_out, _ = self.cross_attn(
            query=lig_bb_tokens, key=rec_tokens, value=rec_tokens,
            key_padding_mask=rec_pad,
        )
        h = self.norm(lig_bb_tokens + cross_out)
        return self.head(h)


class DockingModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers_surf=6, nlayers_bb=4, K=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.rec_surf_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead,
                                               nlayers=nlayers_surf, dropout=dropout)
        self.lig_surf_encoder = SurfaceEncoder(in_dim=6, d_model=d_model, nhead=nhead,
                                               nlayers=nlayers_surf, dropout=dropout)
        self.rec_bb_encoder = BackboneEncoder(d_model=d_model, nhead=nhead,
                                             nlayers=nlayers_bb, dropout=dropout)
        self.lig_bb_encoder = BackboneEncoder(d_model=d_model, nhead=nhead,
                                             nlayers=nlayers_bb, dropout=dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        self.pocket_head = PocketHead(d_model)
        self.binding_head = BindingHead(d_model)
        self.torsion_denoising = TorsionDenoisingHead(d_model)
        self.ss_head = SSHead(d_model=d_model, hidden=128)

    def forward(self, rec_feats, rec_centers, rec_pad,
                lig_feats, lig_centers, lig_pad,
                rec_bb_feats, rec_bb_mask, lig_bb_feats, lig_bb_mask):
        rec_surf_tokens = self.rec_surf_encoder(rec_feats, rec_centers, key_padding_mask=rec_pad)
        lig_surf_tokens = self.lig_surf_encoder(lig_feats, lig_centers, key_padding_mask=lig_pad)

        rec_bb_tokens = self.rec_bb_encoder(rec_bb_feats, mask=~rec_bb_mask)
        lig_bb_tokens = self.lig_bb_encoder(lig_bb_feats, mask=~lig_bb_mask)

        cross_out, _ = self.cross_attn(
            query=rec_surf_tokens, key=lig_surf_tokens, value=lig_surf_tokens,
            key_padding_mask=lig_pad,
        )
        cross_out = self.cross_norm(cross_out)
        valid_r = ~rec_pad
        denom = valid_r.float().sum(dim=1, keepdim=True).clamp(min=1)
        pair_repr = (cross_out * valid_r.unsqueeze(-1).float()).sum(dim=1) / denom

        pocket_center, pocket_radius = self.pocket_head(rec_surf_tokens, rec_pad)
        bind_logit, aff_pred = self.binding_head(
            pair_repr, pocket_center, pocket_radius, lig_centers, lig_pad
        )

        torsion_pred = self.torsion_denoising(lig_bb_tokens, rec_surf_tokens, rec_pad)
        ss_pred = self.ss_head(lig_bb_tokens)

        return {
            'pocket_center': pocket_center, 'pocket_radius': pocket_radius,
            'bind_logit': bind_logit, 'aff_pred': aff_pred,
            'torsion_pred': torsion_pred, 'ss_pred': ss_pred,
            'rec_surf_tokens': rec_surf_tokens, 'lig_surf_tokens': lig_surf_tokens,
            'rec_centers': rec_centers, 'lig_centers': lig_centers,
            'rec_pad': rec_pad, 'lig_pad': lig_pad,
        }


def surface_complementarity_loss(rec_tokens, lig_tokens, rec_centers, lig_centers,
                                 rec_pad, lig_pad, pocket_center, pocket_radius,
                                 contact_thresh=5.0, pocket_extra=2.0):
    device = rec_tokens.device
    B, Tr, D = rec_tokens.shape
    total_loss = 0.0
    used = 0

    for b in range(B):
        vr = ~rec_pad[b]
        vl = ~lig_pad[b]
        r_ctrs = rec_centers[b][vr]
        l_ctrs = lig_centers[b][vl]
        if r_ctrs.size(0) == 0 or l_ctrs.size(0) == 0:
            continue

        pc = pocket_center[b].detach()
        pr = pocket_radius[b, 0].detach()
        dist_p = (r_ctrs - pc.unsqueeze(0)).norm(dim=-1)
        in_pocket = dist_p <= (pr + pocket_extra)
        if in_pocket.sum() == 0:
            in_pocket = torch.ones_like(dist_p, dtype=torch.bool)

        r_tok = rec_tokens[b][vr][in_pocket]
        r_ctrs_p = r_ctrs[in_pocket]
        l_tok = lig_tokens[b][vl]

        with torch.amp.autocast('cuda', enabled=False):
            Dmat = torch.cdist(r_ctrs_p.float(), l_ctrs.float())
        contact = (Dmat <= contact_thresh).float()
        scores = (r_tok @ l_tok.t()) / math.sqrt(D)

        if contact.numel() == 0:
            continue
        pos_frac = contact.mean().clamp(min=1e-4, max=1.0)
        pos_weight = (1.0 - pos_frac) / pos_frac
        bce = F.binary_cross_entropy_with_logits(scores, contact, pos_weight=pos_weight)
        total_loss += bce
        used += 1

    return total_loss / max(1, used) if used > 0 else rec_tokens.new_tensor(0.0)


def load_pretrained_encoders(model, ckpt_path, device):
    """Load Stage 2 pretrained encoder weights into Stage 3 model."""
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Pretrained ckpt not found: {ckpt_path}")
        return
    print(f"[Stage3] Loading pretrained encoders from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"]

    surf_state = {k.replace("surf_encoder.", ""): v for k, v in state.items() if k.startswith("surf_encoder.")}
    if surf_state:
        model.rec_surf_encoder.load_state_dict(surf_state, strict=False)
        model.lig_surf_encoder.load_state_dict(surf_state, strict=False)
        print("  -> Surface encoder weights loaded")

    bb_state = {k.replace("bb_encoder.", ""): v for k, v in state.items() if k.startswith("bb_encoder.")}
    if bb_state:
        model.rec_bb_encoder.load_state_dict(bb_state, strict=False)
        model.lig_bb_encoder.load_state_dict(bb_state, strict=False)
        print("  -> Backbone encoder weights loaded")
