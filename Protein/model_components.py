# -*- coding: utf-8 -*-
"""
Shared neural network components for the backbone-aware protein surface pipeline.
Includes: PointMLP, SurfFormerBlock, SurfaceEncoder, BackboneEncoder, TorsionHead, SSHead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointMLP(nn.Module):
    """Encode K local surface points per patch into a single token."""
    def __init__(self, in_dim=6, hidden=256, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, T, K, _ = x.shape
        h = self.mlp(x.reshape(B * T * K, -1)).reshape(B, T, K, -1)
        h = h.max(dim=2).values
        return self.norm(h)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.shape[1]].unsqueeze(0).to(x.dtype)


class RBFDistanceBias(nn.Module):
    """RBF embedding of pairwise distances for attention bias."""
    def __init__(self, nhead, num_rbf=16):
        super().__init__()
        centers = torch.linspace(0.0, 60.0, num_rbf)
        widths = torch.ones_like(centers) * (centers[1] - centers[0] + 1e-6)
        self.register_buffer('mu', centers)
        self.register_buffer('beta', 1.0 / (2 * widths ** 2))
        self.proj = nn.Linear(num_rbf, nhead)

    def forward(self, centers):
        with torch.no_grad():
            D = torch.cdist(centers, centers)
        diff = D.unsqueeze(-1) - self.mu.view(1, 1, 1, -1)
        rbf = torch.exp(-self.beta.view(1, 1, 1, -1) * diff.pow(2))
        return self.proj(rbf).permute(0, 3, 1, 2)


class SurfFormerBlock(nn.Module):
    """Transformer block with RBF distance bias for surface tokens."""
    def __init__(self, d_model=256, nhead=8, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.bias = RBFDistanceBias(nhead=nhead)
        self.pos = SinusoidalPE(d_model)

    def forward(self, x, centers, key_padding_mask=None):
        B, T, D = x.shape
        x = x + self.pos(x)
        h = self.norm1(x)

        qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_bias = self.bias(centers).to(x.dtype)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        attn_out = attn @ v

        attn_out = attn_out.transpose(2, 1).contiguous().reshape(B, T, D)
        x = x + self.resid_drop(self.proj_out(attn_out))
        x = x + self.ff(self.norm2(x))
        return x


class SurfaceEncoder(nn.Module):
    """Encode surface patches into token representations."""
    def __init__(self, in_dim=6, d_model=256, nhead=8, nlayers=6, dropout=0.1):
        super().__init__()
        self.local = PointMLP(in_dim=in_dim, hidden=d_model, out_dim=d_model)
        self.blocks = nn.ModuleList([
            SurfFormerBlock(d_model=d_model, nhead=nhead, dim_ff=4 * d_model, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, centers, key_padding_mask=None):
        x = self.local(feats)
        for blk in self.blocks:
            x = blk(x, centers, key_padding_mask)
        return self.norm(x)


class BackboneEncoder(nn.Module):
    """Encode residue-level backbone features (N,CA,C coords + torsion sincos)."""
    def __init__(self, d_model=256, nhead=8, nlayers=4, dropout=0.1):
        super().__init__()
        # Input: 9 (N,CA,C relative coords) + 6 (torsion sincos) + 20 (aa one-hot) = 35
        self.input_proj = nn.Sequential(
            nn.Linear(35, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, bb_feats, mask=None):
        x = self.input_proj(bb_feats)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=mask)
        return self.norm(x)


class TorsionHead(nn.Module):
    """Predict torsion angles as sin/cos pairs."""
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 6),  # sin(phi), sin(psi), sin(omega), cos(phi), cos(psi), cos(omega)
        )

    def forward(self, tokens):
        return self.net(tokens)


class SSHead(nn.Module):
    """Predict secondary structure labels (H=0, E=1, C=2)."""
    def __init__(self, d_model=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, tokens):
        return self.net(tokens)


class GumbelCodebook(nn.Module):
    """VQ codebook with Gumbel-Softmax selection."""
    def __init__(self, num_codes=2048, code_dim=256):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim) * 0.02)
        self.num_codes = num_codes

    def forward(self, logits, tau=1.0, hard=True):
        g = -torch.empty_like(logits).exponential_().log()
        y = F.softmax((logits + g) / max(1e-4, tau), dim=-1)
        if hard:
            idx = y.argmax(dim=-1)
            y_hard = F.one_hot(idx, num_classes=logits.shape[-1]).type_as(y)
            y = (y_hard - y).detach() + y
        z = y @ self.codebook
        return z, y


class PatchDecoder(nn.Module):
    """Decode tokens back to K x 3 patch coordinates."""
    def __init__(self, d_model=256, K=50, hidden=512):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, K * 3),
        )

    def forward(self, tokens):
        out = self.mlp(tokens)
        return out.reshape(tokens.shape[0], tokens.shape[1], self.K, 3)
