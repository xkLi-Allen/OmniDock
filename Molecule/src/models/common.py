# -*- coding: utf-8 -*-
"""
Shared neural-network building blocks.
Self-contained - no imports from other projects.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointMLP(nn.Module):
    """
    Local patch encoder: (B, T, K, in_dim) -> (B, T, D) via MLP + max-pool.
    """
    def __init__(self, in_dim: int = 6, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, K, C = x.shape
        h = self.mlp(x.reshape(B * T * K, C))
        h = h.view(B, T, K, -1).max(dim=2).values
        return self.norm(h)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:x.shape[1]].unsqueeze(0).to(x.dtype)


class RBFBias(nn.Module):
    """
    Pairwise Euclidean distance RBF -> per-head additive attention bias.
    Input:  (B, T, 3)  patch centres
    Output: (B, H, T, T)
    """
    def __init__(self, nhead: int, num_rbf: int = 16, max_dist: float = 60.0):
        super().__init__()
        mu   = torch.linspace(0.0, max_dist, num_rbf)
        w    = (max_dist / num_rbf) * torch.ones(num_rbf)
        self.register_buffer("mu",   mu)
        self.register_buffer("beta", 1.0 / (2.0 * w ** 2 + 1e-8))
        self.proj = nn.Linear(num_rbf, nhead, bias=False)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            D = torch.cdist(centers, centers)               # (B, T, T)
        diff = D.unsqueeze(-1) - self.mu.view(1, 1, 1, -1)
        rbf  = torch.exp(-self.beta.view(1, 1, 1, -1) * diff.pow(2))
        return self.proj(rbf).permute(0, 3, 1, 2)          # (B, H, T, T)


class SurfFormerBlock(nn.Module):
    """
    Transformer block with RBF geometric bias and sinusoidal PE.
    Input/output: (B, T, D)
    """
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model  = d_model
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.drop_p   = dropout

        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.rbf_bias = RBFBias(nhead=nhead)
        self.pe       = SinusoidalPE(d_model)

    def forward(self, x: torch.Tensor, centers: torch.Tensor,
                key_padding_mask=None) -> torch.Tensor:
        B, T, D = x.shape
        # add sinusoidal PE
        x = x + self.pe(x)
        # pre-norm MHA
        h   = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # each (B, T, H, dh)
        q = q.permute(0, 2, 1, 3)            # (B, H, T, dh)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        bias = self.rbf_bias(centers).to(x.dtype)  # (B, H, T, T)

        # build additive mask from key_padding_mask
        attn_mask = bias
        if key_padding_mask is not None:
            # key_padding_mask: (B, T), True = ignore
            inf_mask = key_padding_mask.float() * -1e9
            attn_mask = bias + inf_mask.view(B, 1, 1, T)

        try:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.drop_p if self.training else 0.0,
                is_causal=False)
        except TypeError:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + attn_mask
            out    = torch.softmax(scores, dim=-1)
            out    = self.attn_drop(out) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x   = x + self.resid_drop(self.proj_out(out))
        x   = x + self.ff(self.norm2(x))
        return x


class SurfaceEncoder(nn.Module):
    """
    Full surface encoder: PointMLP + N x SurfFormerBlock + LayerNorm.
    Input : feats (B, T, K, 6), centers (B, T, 3)
    Output: (B, T, D)
    """
    def __init__(self, in_dim: int = 6, d_model: int = 256, nhead: int = 8,
                 nlayers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.local  = PointMLP(in_dim=in_dim, hidden=d_model, out_dim=d_model)
        self.blocks = nn.ModuleList([
            SurfFormerBlock(d_model, nhead, 4 * d_model, dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats: torch.Tensor, centers: torch.Tensor,
                key_padding_mask=None) -> torch.Tensor:
        x = self.local(feats)
        for blk in self.blocks:
            x = blk(x, centers, key_padding_mask)
        return self.norm(x)
