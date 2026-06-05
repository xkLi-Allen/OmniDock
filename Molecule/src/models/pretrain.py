# -*- coding: utf-8 -*-
"""
Surface VQ-MAE for Stage-2 self-supervised pretraining.
Self-contained - no imports from other projects.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SurfaceEncoder


class GumbelCodebook(nn.Module):
    """
    Gumbel-Softmax vector quantisation.
    logits -> soft one-hot (B, T, C) -> code vector (B, T, Dc)
    """
    def __init__(self, num_codes: int = 2048, code_dim: int = 256):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim) * 0.02)
        self.num_codes = num_codes

    def forward(self, logits: torch.Tensor, tau: float = 1.0,
                hard: bool = True):
        g  = -torch.empty_like(logits).exponential_().log()
        y  = F.softmax((logits + g) / max(tau, 1e-4), dim=-1)
        if hard:
            idx    = y.argmax(dim=-1)
            y_hard = F.one_hot(idx, self.num_codes).to(y.dtype)
            y      = (y_hard - y).detach() + y
        z = y @ self.codebook
        return z, y


class PatchDecoder(nn.Module):
    """Decode token (D) -> K x 3 relative coordinates."""
    def __init__(self, d_model: int = 256, K: int = 50):
        super().__init__()
        self.K   = K
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, K * 3),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D) -> (B, T, K, 3)
        B, T, D = tokens.shape
        return self.mlp(tokens).view(B, T, self.K, 3)


class CurvatureHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens).squeeze(-1)   # (B, T)


class SurfVQMAE(nn.Module):
    """
    Surface VQ-MAE for self-supervised surface pretraining.

    Forward inputs
    --------------
    feats   : (B, T, K, 6)   local patch features [rel_xyz | normals]
    centers : (B, T, 3)      patch centre positions
    mask    : (B, T)  bool   True = masked patch
    tau     : float          Gumbel temperature
    hard    : bool           straight-through hard Gumbel

    Forward outputs
    ---------------
    rec     : (B, T, K, 3)   reconstructed relative coordinates
    curv    : (B, T)         predicted curvature proxy
    logits  : (B, T, C)      codebook logits (for KL loss)
    post    : (B, T, C)      Gumbel-softmax posterior
    """

    def __init__(
        self,
        in_dim:       int   = 6,
        d_model:      int   = 256,
        nhead:        int   = 8,
        nlayers:      int   = 6,
        K:            int   = 50,
        num_codes:    int   = 2048,
        code_dim:     int   = 256,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.encoder   = SurfaceEncoder(in_dim, d_model, nhead, nlayers, dropout)
        self.pre_norm  = nn.LayerNorm(d_model)
        self.to_logits = nn.Linear(d_model, num_codes)
        self.codebook  = GumbelCodebook(num_codes, code_dim)
        self.up        = nn.Linear(code_dim, d_model) if code_dim != d_model \
                         else nn.Identity()
        self.decoder   = PatchDecoder(d_model, K)
        self.curv_head = CurvatureHead(d_model)

    def forward(self, feats, centers, mask, tau=1.0, hard=True):
        x = self.encoder(feats, centers)        # (B, T, D)
        x = self.pre_norm(x)
        logits = self.to_logits(x)              # (B, T, C)
        zq, post = self.codebook(logits, tau, hard)  # (B, T, Dc), (B, T, C)
        zq = self.up(zq)                        # (B, T, D)
        # replace masked tokens with code vectors
        mf = mask.float().unsqueeze(-1)         # (B, T, 1)
        tokens = x * (1 - mf) + zq * mf
        rec    = self.decoder(tokens)            # (B, T, K, 3)
        curv   = self.curv_head(tokens)          # (B, T)
        return rec, curv, logits, post
