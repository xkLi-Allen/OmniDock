# -*- coding: utf-8 -*-
"""
Stage 4 Model: Torsion-based Backbone Generator.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_components import SurfaceEncoder, TorsionHead


TOPOLOGY_FAMILIES = [
    "helical_peptide",
    "helix_loop_helix",
    "three_helix_bundle",
    "four_helix_bundle",
    "beta_hairpin",
    "coil_peptide",
]


class TorsionGenerator(nn.Module):
    """
    Generates ligand backbone torsion angles conditioned on receptor surface.
    """

    def __init__(self, d_model=256, nhead=8, nlayers_surf=6, nlayers_gen=4,
                 K=50, max_residues=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_residues = max_residues

        self.rec_surf_encoder = SurfaceEncoder(
            in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers_surf, dropout=dropout
        )

        self.pocket_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 4),
        )

        self.hotspot_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.length_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.topology_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, len(TOPOLOGY_FAMILIES)),
        )

        self.torsion_init = nn.Parameter(torch.randn(1, max_residues, d_model) * 0.02)
        self.pos_embed = nn.Embedding(max_residues, d_model)

        self.refine_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            for _ in range(nlayers_gen)
        ])
        self.refine_norm = nn.LayerNorm(d_model)

        self.torsion_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 6),
        )

        self.ss_out = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, rec_feats, rec_centers, rec_pad, num_residues=None):
        B = rec_feats.shape[0]
        device = rec_feats.device

        rec_tokens = self.rec_surf_encoder(rec_feats, rec_centers, key_padding_mask=rec_pad)

        valid = ~rec_pad
        denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (rec_tokens * valid.unsqueeze(-1).float()).sum(dim=1) / denom
        pocket_pred = self.pocket_head(pooled)
        topology_logits = self.topology_head(pooled)
        hotspot_logits = self.hotspot_head(rec_tokens).squeeze(-1).masked_fill(rec_pad, -1e4)

        if num_residues is None:
            length_logit = self.length_head(pooled).squeeze(-1)
            num_residues = int(torch.sigmoid(length_logit).mean().item() * self.max_residues)
            num_residues = max(10, min(num_residues, self.max_residues))

        L = num_residues
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        tgt = self.torsion_init[:, :L, :].expand(B, -1, -1) + self.pos_embed(pos_ids)

        for blk in self.refine_blocks:
            tgt = blk(tgt, rec_tokens, memory_key_padding_mask=rec_pad)
        tgt = self.refine_norm(tgt)

        torsion_sincos = self.torsion_out(tgt)
        sin_vals = torsion_sincos[..., :3]
        cos_vals = torsion_sincos[..., 3:]
        norm = torch.sqrt(sin_vals.pow(2) + cos_vals.pow(2)).clamp(min=1e-6)
        torsion_sincos = torch.cat([sin_vals / norm, cos_vals / norm], dim=-1)
        ss_logits = self.ss_out(tgt)

        return torsion_sincos, ss_logits, pocket_pred, hotspot_logits, topology_logits

    def load_from_stage3(self, stage3_ckpt_path, device='cpu'):
        if not os.path.isfile(stage3_ckpt_path):
            print(f"[WARN] Stage 3 ckpt not found: {stage3_ckpt_path}")
            return
        ckpt = torch.load(stage3_ckpt_path, map_location=device)
        state = ckpt["model"]

        surf_state = {k.replace("rec_surf_encoder.", ""): v
                      for k, v in state.items() if k.startswith("rec_surf_encoder.")}
        if surf_state:
            current = self.rec_surf_encoder.state_dict()
            matched = {k: v for k, v in surf_state.items()
                       if k in current and tuple(current[k].shape) == tuple(v.shape)}
            skipped = len(surf_state) - len(matched)
            self.rec_surf_encoder.load_state_dict(matched, strict=False)
            print(f"[Stage4] Loaded receptor surface encoder from Stage 3: "
                  f"matched={len(matched)}, skipped_shape_mismatch={skipped}")
