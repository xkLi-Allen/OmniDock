# -*- coding: utf-8 -*-
"""
Full protein-ligand docking model for Stage-3.
Self-contained - no imports from other projects.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SurfaceEncoder
from .ligand import LigandEncoder


class PocketHead(nn.Module):
    """
    Predict pocket patch scores from receptor tokens.
    Input : rec_tokens (B, Tr, D), rec_mask (B, Tr)
    Output: pocket_logits (B, Tr),  pocket_center (B,3), pocket_radius (B,1)
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.patch_score = nn.Linear(d_model, 1)
        self.pool_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, rec_tokens, rec_mask):
        pocket_logits = self.patch_score(rec_tokens).squeeze(-1)  # (B, Tr)
        valid  = (~rec_mask).float().unsqueeze(-1)                 # (B, Tr, 1)
        denom  = valid.sum(dim=1).clamp(min=1)                    # (B, 1)
        pooled = (rec_tokens * valid).sum(dim=1) / denom          # (B, D)
        out    = self.pool_mlp(pooled)                            # (B, 4)
        center = out[:, :3]
        radius = F.softplus(out[:, 3:4]) + 1e-6
        return pocket_logits, center, radius


class CrossAttentionModule(nn.Module):
    """
    Cross-attention: receptor tokens attend to ligand atom tokens.
    Input : rec_tokens (B,Tr,D), lig_tokens (B,Na,D), lig_mask (B,Na)
    Output: (B, Tr, D) updated receptor tokens
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rec_tokens, lig_tokens, lig_mask):
        cross, _ = self.attn(
            query=rec_tokens,
            key=lig_tokens,
            value=lig_tokens,
            key_padding_mask=lig_mask,
        )
        return self.norm(rec_tokens + cross)


class MultiTaskHead(nn.Module):
    """
    Predict binding classification and affinity regression.
    Input : pair_repr (B, D), pocket_center (B,3), pocket_radius (B,1)
    Output: bind_logit (B,), affinity (B,)
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        in_dim = d_model + 4
        self.bind_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aff_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, pair_repr, pocket_center, pocket_radius):
        h = torch.cat([pair_repr,
                        pocket_center,
                        pocket_radius], dim=-1)    # (B, D+4)
        bind  = self.bind_head(h).squeeze(-1)      # (B,)
        aff   = self.aff_head(h).squeeze(-1)       # (B,)
        return bind, aff


class DockingModel(nn.Module):
    """
    Full protein-ligand docking model.

    Architecture
    ------------
    1. SurfaceEncoder  : encode receptor surface patches -> rec_tokens (B,Tr,D)
    2. LigandEncoder   : GNN on ligand graph -> lig_tokens (B,Na,D), lig_embed (B,D)
    3. PocketHead      : rec_tokens -> pocket_logits, pocket_center, pocket_radius
    4. CrossAttention  : rec_tokens x lig_tokens -> updated_rec (B,Tr,D)
    5. Pool updated_rec -> pair_repr (B,D)
    6. MultiTaskHead   : pair_repr + pocket info -> bind_logit, affinity

    Forward inputs
    --------------
    rec_feats          : (B, Tr, K, 6)
    rec_centers        : (B, Tr, 3)
    rec_mask           : (B, Tr)  True=pad
    lig_pos            : (B, Na, 3)
    lig_atom_type      : (B, Na)  long
    lig_edge_index_list: list[Tensor(2,Nb)]
    lig_edge_type_list : list[Tensor(Nb,)]
    lig_mask           : (B, Na)  True=pad

    Forward outputs (dict)
    ----------------------
    pocket_logits  : (B, Tr)
    pocket_center  : (B, 3)
    pocket_radius  : (B, 1)
    bind_logit     : (B,)
    affinity_pred  : (B,)
    rec_tokens     : (B, Tr, D)
    lig_tokens     : (B, Na, D)
    rec_centers    : (B, Tr, 3)   (pass-through for loss)
    lig_pos        : (B, Na, 3)   (pass-through for loss)
    rec_mask       : (B, Tr)
    lig_mask       : (B, Na)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 nlayers: int = 6, lig_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.rec_encoder = SurfaceEncoder(
            in_dim=6, d_model=d_model, nhead=nhead,
            nlayers=nlayers, dropout=dropout)
        self.lig_encoder = LigandEncoder(
            d_model=d_model, num_layers=lig_layers, dropout=dropout)
        self.pocket_head = PocketHead(d_model)
        self.cross_attn  = CrossAttentionModule(d_model, nhead, dropout)
        self.task_head   = MultiTaskHead(d_model)

    def forward(self, rec_feats, rec_centers, rec_mask,
                lig_pos, lig_atom_type, lig_edge_index_list,
                lig_edge_type_list, lig_mask):
        # Encode
        rec_tokens = self.rec_encoder(
            rec_feats, rec_centers, rec_mask)          # (B, Tr, D)
        lig_tokens, lig_embed = self.lig_encoder(
            lig_pos, lig_atom_type,
            lig_edge_index_list, lig_edge_type_list,
            lig_mask)                                   # (B,Na,D), (B,D)

        # Pocket prediction (receptor-only)
        pocket_logits, pocket_center, pocket_radius = \
            self.pocket_head(rec_tokens, rec_mask)

        # Cross-attention: rec attends to lig
        updated_rec = self.cross_attn(rec_tokens, lig_tokens, lig_mask)

        # Pool updated receptor tokens -> pair repr
        valid      = (~rec_mask).float().unsqueeze(-1)          # (B,Tr,1)
        denom      = valid.sum(dim=1).clamp(min=1)
        pair_repr  = (updated_rec * valid).sum(dim=1) / denom   # (B,D)

        # Multi-task prediction
        bind_logit, affinity_pred = self.task_head(
            pair_repr, pocket_center, pocket_radius)

        return dict(
            pocket_logits  = pocket_logits,
            pocket_center  = pocket_center,
            pocket_radius  = pocket_radius,
            bind_logit     = bind_logit,
            affinity_pred  = affinity_pred,
            rec_tokens     = rec_tokens,
            lig_tokens     = lig_tokens,
            rec_centers    = rec_centers,
            lig_pos        = lig_pos,
            rec_mask       = rec_mask,
            lig_mask       = lig_mask,
        )
