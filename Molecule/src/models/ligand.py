# -*- coding: utf-8 -*-
"""
Ligand encoder: embed ligand atoms via a lightweight GNN (message-passing).
Self-contained - no imports from other projects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import ELEMENT_LIST


class _MPLayer(nn.Module):
    """One edge-conditioned message-passing step."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.upd = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_feat):
        """
        h          : (Na, D)
        edge_index : (2, Nb)  long
        edge_feat  : (Nb, D)
        """
        if edge_index.shape[1] == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_mlp(
            torch.cat([h[src], h[dst], edge_feat], dim=-1))  # (Nb, D)
        # scatter-add to destination nodes
        agg = torch.zeros_like(h)                            # (Na, D)
        agg.index_add_(0, dst, msg.to(agg.dtype))
        h_new = self.upd(self.drop(agg), h)
        return self.norm(h_new)


class LigandEncoder(nn.Module):
    """
    Two-layer message-passing GNN on the ligand molecular graph.

    Inputs (all batched except edge lists)
    ---------------------------------------
    lig_pos            : (B, Na, 3)
    lig_atom_type      : (B, Na)       long
    lig_edge_index_list: list[Tensor(2,Nb)]  per sample
    lig_edge_type_list : list[Tensor(Nb,)]   per sample
    lig_mask           : (B, Na) bool  True=padding

    Outputs
    -------
    atom_tokens : (B, Na, D)
    graph_embed : (B, D)       mean-pooled (ignoring padding)
    """

    NUM_ELEMENTS   = len(ELEMENT_LIST)
    NUM_BOND_TYPES = 4

    def __init__(self, d_model: int = 256, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.atom_emb = nn.Embedding(self.NUM_ELEMENTS + 1, d_model,
                                     padding_idx=self.NUM_ELEMENTS)
        self.bond_emb = nn.Embedding(self.NUM_BOND_TYPES + 1, d_model,
                                     padding_idx=self.NUM_BOND_TYPES)
        self.pos_mlp  = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.layers   = nn.ModuleList(
            [_MPLayer(d_model, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, lig_pos, lig_atom_type, lig_edge_index_list,
                lig_edge_type_list, lig_mask):
        B, Na, _ = lig_pos.shape
        # initial node features
        h = self.atom_emb(lig_atom_type.clamp(0, self.NUM_ELEMENTS)) \
            + self.pos_mlp(lig_pos)               # (B, Na, D)

        out_list = []
        for b in range(B):
            hb  = h[b]                             # (Na, D)
            ei  = lig_edge_index_list[b].long()    # (2, Nb)
            et  = lig_edge_type_list[b].long().clamp(0, self.NUM_BOND_TYPES)
            ef  = self.bond_emb(et)                # (Nb, D)
            for layer in self.layers:
                hb = layer(hb, ei, ef)
            out_list.append(hb)

        atom_tokens = torch.stack(out_list, dim=0)  # (B, Na, D)
        atom_tokens = self.out_norm(atom_tokens)

        # mean-pool over non-padded atoms
        valid  = (~lig_mask).float().unsqueeze(-1)  # (B, Na, 1)
        denom  = valid.sum(dim=1).clamp(min=1)
        graph_embed = (atom_tokens * valid).sum(dim=1) / denom  # (B, D)

        return atom_tokens, graph_embed
