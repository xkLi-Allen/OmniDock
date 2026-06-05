# -*- coding: utf-8 -*-
"""
Stage-4: Pocket-conditioned 3-D Ligand Diffusion Generator.

Architecture
------------
1. PocketEncoder  : wraps the frozen Stage-2/3 SurfaceEncoder to produce
                    pocket context tokens  (B, Tp, D).
2. AtomDenoiser   : lightweight Transformer that takes noisy atom positions
                    + atom-type embeddings + pocket cross-attention and
                    predicts the denoising direction (epsilon or x0).
3. LigandGenerator: orchestrates DDPM forward/reverse, atom-type sampling,
                    and post-processing to a valid molecule graph.

DDPM schedule
-------------
  T = 500 steps, cosine beta schedule.
  Parametrisation: predict x_0 ("v-prediction" style with clipping).

All code is self-contained (no imports from other inversiondock projects).
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SurfaceEncoder
from ..utils  import ELEMENT_LIST, ELEMENT2IDX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ELEMENTS   = len(ELEMENT_LIST)          # 11
NUM_BOND_TYPES = 4                          # SINGLE / DOUBLE / TRIPLE / AROMATIC
SCAFFOLD_FP_BITS = 256
LIGAND_STAT_NAMES = (
    "scaffold_class",
    "ring_count",
    "aromatic_ring_count",
    "hetero_ring_count",
    "linker_length",
    "substituent_count",
    "hetero_atom_count",
    "halogen_count",
    "branch_count",
    "carbon_count",
    "hetero_ratio_bucket",
)
LIGAND_STAT_BINS = (13, 9, 6, 6, 17, 17, 17, 9, 13, 49, 5)

# ---------------------------------------------------------------------------
# Cosine DDPM schedule
# ---------------------------------------------------------------------------

def _cosine_betas(T: int, s: float = 8e-3) -> torch.Tensor:
    """Return (T,) beta schedule based on cosine annealing."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0, 0.999).float()


class DDPMSchedule(nn.Module):
    """Pre-computes and registers all DDPM noise-schedule buffers."""

    def __init__(self, T: int = 500):
        super().__init__()
        self.T = T
        betas = _cosine_betas(T)                        # (T,)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)        # (T,)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)

        self.register_buffer("betas",         betas)
        self.register_buffer("alphas",        alphas)
        self.register_buffer("alpha_bar",     alpha_bar)
        self.register_buffer("alpha_bar_prev",alpha_bar_prev)
        self.register_buffer("sqrt_ab",       alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_ab", (1.0 - alpha_bar).sqrt())
        # posterior variance q(x_{t-1}|x_t, x_0)
        post_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp(min=1e-8)
        self.register_buffer("post_var", post_var)
        self.register_buffer("post_log_var_clipped",
                             torch.log(post_var.clamp(min=1e-20)))
        self.register_buffer("post_mean_coef1",
                             betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar).clamp(min=1e-8))
        self.register_buffer("post_mean_coef2",
                             (1.0 - alpha_bar_prev) * alphas.sqrt() /
                             (1.0 - alpha_bar).clamp(min=1e-8))

    # ---- forward process ------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x0: (B,N,3), t: (B,) long -> x_t: (B,N,3)"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.sqrt_ab[t].view(-1, 1, 1)
        oab = self.sqrt_one_minus_ab[t].view(-1, 1, 1)
        return ab * x0 + oab * noise

    # ---- reverse step (DDPM ancestral sampling) -------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_idx: int,
                 x0_pred: torch.Tensor) -> torch.Tensor:
        """One reverse step: x_t -> x_{t-1} using predicted x_0."""
        t = torch.full((x_t.shape[0],), t_idx, device=x_t.device, dtype=torch.long)
        mu = (self.post_mean_coef1[t].view(-1, 1, 1) * x0_pred
              + self.post_mean_coef2[t].view(-1, 1, 1) * x_t)
        if t_idx == 0:
            return mu
        log_var = self.post_log_var_clipped[t].view(-1, 1, 1)
        noise   = torch.randn_like(x_t)
        return mu + (0.5 * log_var).exp() * noise


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,)
        half = self.d // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, device=t.device) / half)
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, d)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Atom Denoiser Transformer
# ---------------------------------------------------------------------------

class AtomDenoiserBlock(nn.Module):
    """Self-attn + cross-attn (to pocket tokens) + FFN."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead,
                                                dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead,
                                                dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                pocket_tokens: torch.Tensor,
                pocket_mask:   Optional[torch.Tensor] = None) -> torch.Tensor:
        # self-attention over atoms
        h, _ = self.self_attn(x, x, x)
        x = self.norm1(x + h)
        # cross-attention: atoms attend to pocket
        h, _ = self.cross_attn(x, pocket_tokens, pocket_tokens,
                               key_padding_mask=pocket_mask)
        x = self.norm2(x + h)
        # FFN
        x = self.norm3(x + self.ff(x))
        return x


class AtomDenoiser(nn.Module):
    """
    Predicts x_0 given noisy x_t, atom types, timestep, and pocket context.

    Inputs
    ------
    x_t          : (B, Na, 3)   noisy atom positions
    atom_type    : (B, Na)      long  (ELEMENT_LIST indices)
    t            : (B,)         long  timestep
    pocket_tokens: (B, Tp, D)   from SurfaceEncoder
    pocket_mask  : (B, Tp)      True = padding
    lig_mask     : (B, Na)      True = padding (unused atoms)

    Output
    ------
    x0_pred : (B, Na, 3)  predicted denoised positions
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 nlayers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.atom_emb  = nn.Embedding(NUM_ELEMENTS + 1, d_model,
                                      padding_idx=NUM_ELEMENTS)
        self.pos_proj  = nn.Linear(3, d_model)
        self.t_emb     = TimestepEmbedding(d_model)
        self.t_proj    = nn.Linear(d_model, d_model)
        self.blocks    = nn.ModuleList([
            AtomDenoiserBlock(d_model, nhead, dropout)
            for _ in range(nlayers)
        ])
        self.out_norm  = nn.LayerNorm(d_model)
        self.out_proj  = nn.Linear(d_model, 3)   # predict x_0 (coords)

    def forward(self, x_t: torch.Tensor, atom_type: torch.Tensor,
                t: torch.Tensor,
                pocket_tokens: torch.Tensor,
                pocket_mask: Optional[torch.Tensor] = None,
                lig_mask:    Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Na, _ = x_t.shape
        # node features: position + atom type + time
        h = self.pos_proj(x_t) + \
            self.atom_emb(atom_type.clamp(0, NUM_ELEMENTS)) + \
            self.t_proj(self.t_emb(t)).unsqueeze(1)        # (B, Na, D)

        for blk in self.blocks:
            h = blk(h, pocket_tokens, pocket_mask)

        h = self.out_norm(h)
        delta = self.out_proj(h)                            # (B, Na, 3)

        # mask padding atoms
        if lig_mask is not None:
            delta = delta.masked_fill(lig_mask.unsqueeze(-1), 0.0)
        return x_t + delta                                  # residual -> x0


# ---------------------------------------------------------------------------
# Atom-type prediction head
# ---------------------------------------------------------------------------

class AtomTypeHead(nn.Module):
    """
    From final hidden state (B, Na, D) -> atom type logits (B, Na, E).
    Trained with cross-entropy against ground-truth element labels.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, NUM_ELEMENTS),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B, Na, NUM_ELEMENTS)


# ---------------------------------------------------------------------------
# Bond prediction head
# ---------------------------------------------------------------------------

class BondHead(nn.Module):
    """
    Predicts bond type between all atom pairs that are within a distance cutoff.
    (B, Na, D) -> (B, Na, Na, NUM_BOND_TYPES+1)  (+1 for 'no bond').
    Applied after generation; used for SDF reconstruction.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, NUM_BOND_TYPES + 1),
        )

    def forward(self, h: torch.Tensor, lig_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B, Na, D = h.shape
        hi = h.unsqueeze(2).expand(B, Na, Na, D)
        hj = h.unsqueeze(1).expand(B, Na, Na, D)
        logits = self.net(torch.cat([hi, hj], dim=-1))   # (B, Na, Na, 5)
        if lig_mask is not None:
            # mask rows and cols corresponding to padding
            m = lig_mask.unsqueeze(2) | lig_mask.unsqueeze(1)
            logits = logits.masked_fill(m.unsqueeze(-1), float("-inf"))
        return logits


class DegreeHead(nn.Module):
    """Predict per-atom heavy-atom graph degree buckets 0..4."""

    def __init__(self, d_model: int = 256, max_degree: int = 4):
        super().__init__()
        self.max_degree = max_degree
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, max_degree + 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class DirectLigandHead(nn.Module):
    """Receptor-only ligand predictor for small-data supervised fitting.

    Uses learned atom-slot queries that cross-attend to pocket tokens and
    directly predicts local ligand coordinates, atom types and bonds without
    conditioning on ground-truth ligand atom types during generation.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 nlayers: int = 3, max_atoms: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.max_atoms = max_atoms
        self.atom_query = nn.Embedding(max_atoms, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.type_head = AtomTypeHead(d_model)
        self.bond_head = BondHead(d_model)
        self.degree_head = DegreeHead(d_model)

    def forward(self, pocket_tokens: torch.Tensor,
                rec_mask: Optional[torch.Tensor] = None,
                num_atoms: Optional[int] = None) -> dict:
        B = pocket_tokens.shape[0]
        Na = self.max_atoms if num_atoms is None else min(num_atoms, self.max_atoms)
        idx = torch.arange(Na, device=pocket_tokens.device)
        q = self.atom_query(idx).unsqueeze(0).expand(B, Na, -1)
        h = self.decoder(q, pocket_tokens, memory_key_padding_mask=rec_mask)
        h = self.norm(h)
        return dict(
            positions_rel=self.pos_head(h),
            type_logits=self.type_head(h),
            bond_logits=self.bond_head(h),
            degree_logits=self.degree_head(h),
            h_final=h,
        )


# ---------------------------------------------------------------------------
# Pocket encoder wrapper (thin wrapper around SurfaceEncoder)
# ---------------------------------------------------------------------------

class PocketEncoder(nn.Module):
    """
    Thin wrapper so Stage-4 can either:
      (a) load pre-trained Stage-2/3 weights into self.encoder, or
      (b) train from scratch alongside the denoiser.

    Input : rec_feats (B, Tp, K, 6), rec_centers (B, Tp, 3),
            rec_mask  (B, Tp) True=pad
    Output: pocket_tokens (B, Tp, D)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 nlayers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.encoder = SurfaceEncoder(
            in_dim=6, d_model=d_model, nhead=nhead,
            nlayers=nlayers, dropout=dropout)

    def forward(self, rec_feats: torch.Tensor,
                rec_centers: torch.Tensor,
                rec_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(rec_feats, rec_centers, rec_mask)


# ---------------------------------------------------------------------------
# Main LigandGenerator
# ---------------------------------------------------------------------------

class LigandGenerator(nn.Module):
    """
    Pocket-conditioned 3-D ligand DDPM generator.

    Training
    --------
    loss = coord_loss (MSE x_0 prediction) + atom_type_ce + bond_ce

    Inference
    ---------
    Given rec_feats / rec_centers from Stage-1 .npz (or freshly computed
    from a PDB), run DDPM reverse from pure noise for T steps, then
    predict atom types and bond types from the final hidden states.

    Parameters
    ----------
    d_model        : transformer width
    nhead          : attention heads
    enc_nlayers    : SurfaceEncoder depth  (for pocket)
    den_nlayers    : AtomDenoiser depth
    T              : diffusion timesteps
    max_atoms      : maximum atoms to generate per molecule
    dropout        : dropout rate
    """

    def __init__(
        self,
        d_model:     int   = 256,
        nhead:       int   = 8,
        enc_nlayers: int   = 6,
        den_nlayers: int   = 4,
        T:           int   = 500,
        max_atoms:   int   = 64,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.T         = T
        self.max_atoms = max_atoms

        self.pocket_enc = PocketEncoder(d_model, nhead, enc_nlayers, dropout)
        self.schedule   = DDPMSchedule(T)
        self.denoiser   = AtomDenoiser(d_model, nhead, den_nlayers, dropout)
        self.type_head  = AtomTypeHead(d_model)
        self.bond_head  = BondHead(d_model)
        self.direct_head = DirectLigandHead(
            d_model=d_model,
            nhead=nhead,
            nlayers=max(1, den_nlayers),
            max_atoms=max_atoms,
            dropout=dropout,
        )

        # exact atom-count classifier: 1,2,...,max_atoms.  This is important
        # for receptor-only RMSD evaluation because generation cannot rely on
        # a ground-truth ligand atom count at inference time.
        self._atom_buckets = list(range(1, max_atoms + 1))
        self.natom_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, len(self._atom_buckets)),
        )
        self.graph_count_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.ligand_stat_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, bins),
            )
            for bins in LIGAND_STAT_BINS
        ])
        self.scaffold_token_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.scaffold_fp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, SCAFFOLD_FP_BITS),
        )
        self.degree_hist_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 5),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pocket_pool(self, pocket_tokens: torch.Tensor,
                     pocket_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean-pool valid pocket tokens -> (B, D)."""
        valid = (~pocket_mask).float().unsqueeze(-1) \
            if pocket_mask is not None \
            else torch.ones(*pocket_tokens.shape[:2], 1, device=pocket_tokens.device)
        denom = valid.sum(1).clamp(min=1)
        return (pocket_tokens * valid).sum(1) / denom   # (B, D)

    def _encode_pocket(
        self,
        rec_feats:   torch.Tensor,
        rec_centers: torch.Tensor,
        rec_mask:    Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns pocket_tokens (B,Tp,D) and pocket_pooled (B,D)."""
        pt = self.pocket_enc(rec_feats, rec_centers, rec_mask)
        pp = self._pocket_pool(pt, rec_mask)
        return pt, pp

    def _direct_geometry_losses(
        self,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        lig_mask: Optional[torch.Tensor],
        edge_index_list: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        B, Na, _ = pred_pos.shape
        device = pred_pos.device
        valid = ~lig_mask if lig_mask is not None else torch.ones(B, Na, dtype=torch.bool, device=device)
        pred_d = torch.cdist(pred_pos, pred_pos).clamp(min=1e-6)
        true_d = torch.cdist(true_pos, true_pos).clamp(min=1e-6)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
        eye = torch.eye(Na, dtype=torch.bool, device=device).unsqueeze(0)
        pair_valid = pair_valid & (~eye)

        if pair_valid.any():
            dist_loss = F.smooth_l1_loss(pred_d[pair_valid], true_d[pair_valid])
            clash_loss = F.relu(0.85 - pred_d[pair_valid]).pow(2).mean()
        else:
            dist_loss = pred_pos.new_tensor(0.0)
            clash_loss = pred_pos.new_tensor(0.0)

        pred_centered = pred_pos - (pred_pos * valid.float().unsqueeze(-1)).sum(1, keepdim=True) / valid.float().sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        true_centered = true_pos - (true_pos * valid.float().unsqueeze(-1)).sum(1, keepdim=True) / valid.float().sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        pred_rg = ((pred_centered.pow(2).sum(-1) * valid.float()).sum(1) / valid.float().sum(1).clamp(min=1)).sqrt()
        true_rg = ((true_centered.pow(2).sum(-1) * valid.float()).sum(1) / valid.float().sum(1).clamp(min=1)).sqrt()
        radius_loss = F.smooth_l1_loss(pred_rg, true_rg)

        edge_len_loss = pred_pos.new_tensor(0.0)
        n_edge_batches = 0
        if edge_index_list is not None:
            for b in range(B):
                ei = edge_index_list[b]
                if ei is None or ei.numel() == 0:
                    continue
                ei = ei.long().to(device)
                src, dst = ei[0], ei[1]
                keep = (src < Na) & (dst < Na) & valid[b, src] & valid[b, dst]
                src, dst = src[keep], dst[keep]
                if src.numel() == 0:
                    continue
                keep_unique = src < dst
                src, dst = src[keep_unique], dst[keep_unique]
                if src.numel() == 0:
                    continue
                pred_len = (pred_pos[b, src] - pred_pos[b, dst]).norm(dim=-1)
                true_len = (true_pos[b, src] - true_pos[b, dst]).norm(dim=-1)
                edge_len_loss = edge_len_loss + F.smooth_l1_loss(pred_len, true_len)
                n_edge_batches += 1
        edge_len_loss = edge_len_loss / max(n_edge_batches, 1)
        return dict(
            dist_loss=dist_loss,
            clash_loss=clash_loss,
            radius_loss=radius_loss,
            edge_len_loss=edge_len_loss,
        )

    @torch.no_grad()
    def _spread_direct_positions(self, positions_rel: torch.Tensor) -> torch.Tensor:
        B, Na, _ = positions_rel.shape
        if Na <= 1:
            return positions_rel
        centered = positions_rel - positions_rel.mean(dim=1, keepdim=True)
        rg = centered.pow(2).sum(-1).mean(1).sqrt()
        min_rg = (0.55 * (Na ** (1.0 / 3.0))).to(centered.device) if isinstance(Na, torch.Tensor) else 0.55 * (Na ** (1.0 / 3.0))
        scale = torch.clamp(torch.as_tensor(min_rg, dtype=centered.dtype, device=centered.device) / rg.clamp(min=1e-4), min=1.0, max=8.0)
        centered = centered * scale.view(B, 1, 1)
        eye = torch.eye(Na, dtype=torch.bool, device=centered.device).unsqueeze(0)
        for _ in range(8):
            diff = centered.unsqueeze(2) - centered.unsqueeze(1)
            d = diff.norm(dim=-1).clamp(min=1e-6)
            close = (d < 0.95) & (~eye)
            if not close.any():
                break
            direction = diff / d.unsqueeze(-1)
            push = direction * ((0.95 - d).clamp(min=0.0) * close.float()).unsqueeze(-1)
            centered = centered + 0.18 * push.sum(dim=2)
            centered = centered - centered.mean(dim=1, keepdim=True)
        return centered

    def _ligand_stat_losses(
        self,
        pocket_pooled: torch.Tensor,
        ligand_stats: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits = [head(pocket_pooled) for head in self.ligand_stat_heads]
        if ligand_stats is None:
            zero = pocket_pooled.new_tensor(0.0)
            return zero, logits
        losses = []
        for i, logit in enumerate(logits):
            target = ligand_stats[:, i].long().to(logit.device).clamp(0, logit.shape[-1] - 1)
            losses.append(F.cross_entropy(logit, target))
        return torch.stack(losses).mean(), logits

    def _scaffold_token_targets(
        self,
        lig_atom_type: torch.Tensor,
        lig_mask: Optional[torch.Tensor],
        ligand_stats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, Na = lig_atom_type.shape
        target = torch.full((B, Na), 2, dtype=torch.long, device=lig_atom_type.device)
        if ligand_stats is None:
            return target
        ring_counts = ligand_stats[:, 1].long().to(lig_atom_type.device).clamp(min=0)
        for b in range(B):
            valid = (~lig_mask[b]) if lig_mask is not None else torch.ones(Na, dtype=torch.bool, device=lig_atom_type.device)
            n_valid = int(valid.sum().item())
            if n_valid <= 0:
                continue
            n_ring_atoms = int(min(n_valid, max(0, ring_counts[b].item()) * 6))
            if n_ring_atoms > 0:
                target[b, :n_ring_atoms] = 0
            n_linker = int(min(max(n_valid - n_ring_atoms, 0), ligand_stats[b, 4].item() if ligand_stats is not None else 0))
            if n_linker > 0:
                target[b, n_ring_atoms:n_ring_atoms + n_linker] = 1
        return target

    def _scaffold_fp_loss(
        self,
        pocket_pooled: torch.Tensor,
        scaffold_fp: Optional[torch.Tensor],
        scaffold_fp_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.scaffold_fp_head(pocket_pooled)
        if scaffold_fp is None:
            return pocket_pooled.new_tensor(0.0), logits
        target = scaffold_fp.to(logits.device, dtype=logits.dtype)
        loss_per_bit = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss_per_sample = loss_per_bit.mean(dim=-1)
        if scaffold_fp_mask is not None:
            mask = scaffold_fp_mask.to(logits.device, dtype=logits.dtype).view(-1)
            denom = mask.sum().clamp(min=1.0)
            loss = (loss_per_sample * mask).sum() / denom
        else:
            loss = loss_per_sample.mean()
        return loss, logits

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rec_feats:    torch.Tensor,           # (B, Tp, K, 6)
        rec_centers:  torch.Tensor,           # (B, Tp, 3)
        rec_mask:     Optional[torch.Tensor], # (B, Tp)
        lig_pos:      torch.Tensor,           # (B, Na, 3)  ground-truth (pocket-centred)
        lig_atom_type:torch.Tensor,           # (B, Na)     long
        lig_mask:     Optional[torch.Tensor] = None,  # (B, Na)
        lig_edge_index_list: Optional[List[torch.Tensor]] = None,
        lig_edge_type_list:  Optional[List[torch.Tensor]] = None,
        ligand_stats: Optional[torch.Tensor] = None,
        scaffold_fp: Optional[torch.Tensor] = None,
        scaffold_fp_mask: Optional[torch.Tensor] = None,
        pocket_center: Optional[torch.Tensor] = None,  # (B, 3) unused in fwd, kept for API consistency
        direct: bool = False,
    ) -> dict:
        """
        Training forward: add noise, denoise, compute losses.

        Returns dict with keys:
          loss         : scalar total loss
          coord_loss   : MSE on x_0
          type_loss    : atom-type CE
          bond_loss    : bond-type CE
          natom_loss   : num-atom CE
        """
        B, Na, _ = lig_pos.shape
        device   = lig_pos.device

        # ---- encode pocket ----
        pocket_tokens, pocket_pooled = self._encode_pocket(
            rec_feats, rec_centers, rec_mask)

        if direct:
            direct_out = self.direct_head(pocket_tokens, rec_mask, num_atoms=Na)
            x0_pred = direct_out["positions_rel"]
            type_logits = direct_out["type_logits"]
            bond_logits = direct_out["bond_logits"]
            degree_logits = direct_out["degree_logits"]

            if lig_mask is not None:
                valid = (~lig_mask).float().unsqueeze(-1)
                coord_loss = (F.mse_loss(x0_pred, lig_pos, reduction='none')
                              * valid).sum() / valid.sum().clamp(min=1)
                valid_flat = (~lig_mask).reshape(-1)
                type_loss = F.cross_entropy(
                    type_logits.reshape(-1, NUM_ELEMENTS)[valid_flat],
                    lig_atom_type.reshape(-1)[valid_flat].clamp(0, NUM_ELEMENTS - 1),
                )
            else:
                coord_loss = F.mse_loss(x0_pred, lig_pos)
                type_loss = F.cross_entropy(
                    type_logits.reshape(-1, NUM_ELEMENTS),
                    lig_atom_type.reshape(-1).clamp(0, NUM_ELEMENTS - 1),
                )

            bond_loss = lig_pos.new_tensor(0.0)
            degree_target = torch.zeros((B, Na), dtype=torch.long, device=device)
            bond_count_target = torch.zeros((B,), dtype=torch.float32, device=device)
            ring_count_target = torch.zeros((B,), dtype=torch.float32, device=device)
            branch_count_target = torch.zeros((B,), dtype=torch.float32, device=device)
            degree_hist_target = torch.zeros((B, 5), dtype=torch.float32, device=device)
            if lig_edge_index_list is not None:
                for b in range(B):
                    valid_atoms = (~lig_mask[b]).nonzero(as_tuple=False).view(-1) \
                        if lig_mask is not None else torch.arange(Na, device=device)
                    if int(valid_atoms.numel()) < 2:
                        continue
                    target = torch.zeros((Na, Na), dtype=torch.long, device=device)
                    edge_mask = torch.zeros((Na, Na), dtype=torch.bool, device=device)
                    ei = lig_edge_index_list[b].long().to(device) if lig_edge_index_list[b] is not None else None
                    et = lig_edge_type_list[b].long().to(device) if lig_edge_type_list is not None else None
                    if ei is not None and ei.shape[1] > 0:
                        src, dst = ei[0], ei[1]
                        keep = (src < Na) & (dst < Na)
                        src, dst = src[keep], dst[keep]
                        et = et[keep].clamp(1, NUM_BOND_TYPES) if et is not None else torch.ones_like(src)
                        target[src, dst] = et
                        target[dst, src] = et
                        edge_mask[src, dst] = True
                        edge_mask[dst, src] = True
                        unique = src < dst
                        degree_target[b].scatter_add_(0, src[unique], torch.ones_like(src[unique], dtype=torch.long))
                        degree_target[b].scatter_add_(0, dst[unique], torch.ones_like(dst[unique], dtype=torch.long))
                        n_edges = int(unique.sum().item())
                        n_atoms_b = int(valid_atoms.numel())
                        bond_count_target[b] = float(n_edges)
                        ring_count_target[b] = float(max(0, n_edges - n_atoms_b + 1))
                        deg_valid = degree_target[b, valid_atoms].clamp(0, 4)
                        branch_count_target[b] = float((deg_valid >= 3).sum().item())
                        degree_hist_target[b].scatter_add_(0, deg_valid, torch.ones_like(deg_valid, dtype=torch.float32))
                        degree_hist_target[b] = degree_hist_target[b] / max(n_atoms_b, 1)
                    vv_i, vv_j = torch.meshgrid(valid_atoms, valid_atoms, indexing="ij")
                    pair_mask = vv_i < vv_j
                    pos_pair_mask = edge_mask[vv_i, vv_j] & pair_mask
                    neg_pair_mask = (~edge_mask[vv_i, vv_j]) & pair_mask
                    pos_i = vv_i[pos_pair_mask]
                    pos_j = vv_j[pos_pair_mask]
                    neg_i_all = vv_i[neg_pair_mask]
                    neg_j_all = vv_j[neg_pair_mask]
                    n_pos = int(pos_i.numel())
                    max_neg = max(n_pos * 3, 32)
                    if neg_i_all.numel() > max_neg:
                        perm = torch.randperm(neg_i_all.numel(), device=device)[:max_neg]
                        neg_i, neg_j = neg_i_all[perm], neg_j_all[perm]
                    else:
                        neg_i, neg_j = neg_i_all, neg_j_all
                    if n_pos == 0 and neg_i.numel() == 0:
                        continue
                    pi = torch.cat([pos_i, neg_i], dim=0)
                    pj = torch.cat([pos_j, neg_j], dim=0)
                    y = target[pi, pj]
                    bond_loss = bond_loss + F.cross_entropy(bond_logits[b, pi, pj, :], y)
                bond_loss = bond_loss / max(B, 1)

            if lig_mask is not None:
                valid_flat = (~lig_mask).reshape(-1)
                degree_loss = F.cross_entropy(
                    degree_logits.reshape(-1, degree_logits.shape[-1])[valid_flat],
                    degree_target.clamp(0, degree_logits.shape[-1] - 1).reshape(-1)[valid_flat],
                )
            else:
                degree_loss = F.cross_entropy(
                    degree_logits.reshape(-1, degree_logits.shape[-1]),
                    degree_target.clamp(0, degree_logits.shape[-1] - 1).reshape(-1),
                )

            geom = self._direct_geometry_losses(
                x0_pred, lig_pos, lig_mask, lig_edge_index_list)

            natom_logits = self.natom_head(pocket_pooled)
            if lig_mask is not None:
                num_atoms = (~lig_mask).sum(dim=1).long()
            else:
                num_atoms = torch.full((B,), Na, device=device, dtype=torch.long)
            buckets = torch.tensor(self._atom_buckets, device=device)
            bucket_idx = (num_atoms.unsqueeze(1) - buckets.unsqueeze(0)).abs().argmin(dim=1)
            natom_weight = torch.ones(len(self._atom_buckets), dtype=lig_pos.dtype, device=device)
            natom_weight[:8] = 1.4
            natom_weight[8:41] = 2.0
            natom_weight[41:] = 0.6
            natom_loss = F.cross_entropy(natom_logits, bucket_idx, weight=natom_weight)

            graph_counts = F.softplus(self.graph_count_head(pocket_pooled))
            ligand_stat_loss, ligand_stat_logits = self._ligand_stat_losses(pocket_pooled, ligand_stats)
            scaffold_fp_loss, scaffold_fp_logits = self._scaffold_fp_loss(pocket_pooled, scaffold_fp, scaffold_fp_mask)
            scaffold_token_logits = self.scaffold_token_head(direct_out["h_final"])
            scaffold_token_target = self._scaffold_token_targets(lig_atom_type, lig_mask, ligand_stats)
            if lig_mask is not None:
                scaffold_valid = (~lig_mask).reshape(-1)
                scaffold_token_loss = F.cross_entropy(
                    scaffold_token_logits.reshape(-1, 3)[scaffold_valid],
                    scaffold_token_target.reshape(-1)[scaffold_valid],
                )
            else:
                scaffold_token_loss = F.cross_entropy(scaffold_token_logits.reshape(-1, 3), scaffold_token_target.reshape(-1))
            graph_scale = num_atoms.float().clamp(min=1.0)
            bond_count_loss = F.smooth_l1_loss(graph_counts[:, 0] / graph_scale, bond_count_target / graph_scale)
            ring_count_loss = F.smooth_l1_loss(graph_counts[:, 1] / 4.0, ring_count_target.clamp(max=4) / 4.0)
            branch_count_loss = F.smooth_l1_loss(graph_counts[:, 2] / 8.0, branch_count_target.clamp(max=8) / 8.0)
            degree_hist_logits = self.degree_hist_head(pocket_pooled)
            degree_hist_loss = F.kl_div(
                F.log_softmax(degree_hist_logits, dim=-1),
                degree_hist_target.clamp(min=1e-4) / degree_hist_target.clamp(min=1e-4).sum(dim=-1, keepdim=True),
                reduction="batchmean",
            )

            loss = (
                1.5 * coord_loss
                + 1.0 * type_loss
                + 1.0 * bond_loss
                + 1.0 * degree_loss
                + 2.0 * natom_loss
                + 0.8 * bond_count_loss
                + 0.8 * ring_count_loss
                + 0.6 * branch_count_loss
                + 0.8 * ligand_stat_loss
                + 0.5 * scaffold_token_loss
                + 0.3 * scaffold_fp_loss
                + 0.6 * degree_hist_loss
                + 1.5 * geom["dist_loss"]
                + 1.0 * geom["edge_len_loss"]
                + 1.0 * geom["radius_loss"]
                + 2.0 * geom["clash_loss"]
            )
            return dict(
                loss=loss,
                coord_loss=coord_loss,
                type_loss=type_loss,
                bond_loss=bond_loss,
                natom_loss=natom_loss,
                degree_loss=degree_loss,
                bond_count_loss=bond_count_loss,
                ring_count_loss=ring_count_loss,
                branch_count_loss=branch_count_loss,
                ligand_stat_loss=ligand_stat_loss,
                scaffold_token_loss=scaffold_token_loss,
                scaffold_fp_loss=scaffold_fp_loss,
                ligand_stat_logits=ligand_stat_logits,
                scaffold_token_logits=scaffold_token_logits,
                scaffold_fp_logits=scaffold_fp_logits,
                degree_hist_loss=degree_hist_loss,
                dist_loss=geom["dist_loss"],
                edge_len_loss=geom["edge_len_loss"],
                radius_loss=geom["radius_loss"],
                clash_loss=geom["clash_loss"],
                x0_pred=x0_pred,
            )

        # ---- sample timestep and add noise ----
        t = torch.randint(0, self.T, (B,), device=device)
        noise   = torch.randn_like(lig_pos)
        x_t     = self.schedule.q_sample(lig_pos, t, noise)

        # ---- denoise: predict x_0 ----
        x0_pred = self.denoiser(x_t, lig_atom_type, t,
                                pocket_tokens, rec_mask, lig_mask)

        # ---- coordinate loss ----
        if lig_mask is not None:
            valid = (~lig_mask).float().unsqueeze(-1)   # (B, Na, 1)
            coord_loss = (F.mse_loss(x0_pred, lig_pos, reduction='none')
                          * valid).sum() / valid.sum().clamp(min=1)
        else:
            coord_loss = F.mse_loss(x0_pred, lig_pos)

        # ---- atom type loss ----
        # Re-run denoiser at t=0 (clean) to get hidden state for type head
        with torch.no_grad():
            t0 = torch.zeros(B, device=device, dtype=torch.long)
        h_clean = self.denoiser.pos_proj(lig_pos) + \
                  self.denoiser.atom_emb(
                      lig_atom_type.clamp(0, NUM_ELEMENTS)) + \
                  self.denoiser.t_proj(self.denoiser.t_emb(t0)).unsqueeze(1)
        for blk in self.denoiser.blocks:
            h_clean = blk(h_clean, pocket_tokens, rec_mask)
        h_clean = self.denoiser.out_norm(h_clean)
        type_logits = self.type_head(h_clean)   # (B, Na, E)

        if lig_mask is not None:
            valid_flat = (~lig_mask).reshape(-1)  # (B*Na,)
            type_loss = F.cross_entropy(
                type_logits.reshape(-1, NUM_ELEMENTS)[valid_flat],
                lig_atom_type.reshape(-1)[valid_flat].clamp(0, NUM_ELEMENTS - 1),
            )
        else:
            type_loss = F.cross_entropy(
                type_logits.reshape(-1, NUM_ELEMENTS),
                lig_atom_type.reshape(-1).clamp(0, NUM_ELEMENTS - 1),
            )

        # ---- bond loss ----
        bond_loss = lig_pos.new_tensor(0.0)
        if lig_edge_index_list is not None:
            bond_logits = self.bond_head(h_clean, lig_mask)  # (B,Na,Na,5)
            for b in range(B):
                valid_atoms = (~lig_mask[b]).nonzero(as_tuple=False).view(-1) \
                    if lig_mask is not None else torch.arange(Na, device=device)
                nv = int(valid_atoms.numel())
                if nv < 2:
                    continue

                target = torch.zeros((Na, Na), dtype=torch.long, device=device)
                edge_mask = torch.zeros((Na, Na), dtype=torch.bool, device=device)

                ei = lig_edge_index_list[b].long().to(device) if lig_edge_index_list[b] is not None else None
                et = lig_edge_type_list[b].long().to(device) if lig_edge_type_list is not None else None
                if ei is not None and ei.shape[1] > 0:
                    src, dst = ei[0], ei[1]
                    keep = (src < Na) & (dst < Na)
                    src, dst = src[keep], dst[keep]
                    et = et[keep].clamp(1, NUM_BOND_TYPES) if et is not None else torch.ones_like(src)
                    target[src, dst] = et
                    target[dst, src] = et
                    edge_mask[src, dst] = True
                    edge_mask[dst, src] = True

                vv_i, vv_j = torch.meshgrid(valid_atoms, valid_atoms, indexing="ij")
                pair_mask = vv_i < vv_j
                pos_pair_mask = edge_mask[vv_i, vv_j] & pair_mask
                neg_pair_mask = (~edge_mask[vv_i, vv_j]) & pair_mask

                pos_i = vv_i[pos_pair_mask]
                pos_j = vv_j[pos_pair_mask]
                neg_i_all = vv_i[neg_pair_mask]
                neg_j_all = vv_j[neg_pair_mask]

                n_pos = int(pos_i.numel())
                max_neg = max(n_pos * 3, 32)
                if neg_i_all.numel() > max_neg:
                    perm = torch.randperm(neg_i_all.numel(), device=device)[:max_neg]
                    neg_i = neg_i_all[perm]
                    neg_j = neg_j_all[perm]
                else:
                    neg_i, neg_j = neg_i_all, neg_j_all

                if n_pos == 0 and neg_i.numel() == 0:
                    continue
                pi = torch.cat([pos_i, neg_i], dim=0)
                pj = torch.cat([pos_j, neg_j], dim=0)
                y = target[pi, pj]
                logits_b = bond_logits[b, pi, pj, :]
                bond_loss = bond_loss + F.cross_entropy(logits_b, y)
            bond_loss = bond_loss / max(B, 1)

        # ---- num-atom and ligand scaffold/stat losses ----
        natom_logits = self.natom_head(pocket_pooled)        # (B, n_buckets)
        if lig_mask is not None:
            num_atoms = (~lig_mask).sum(dim=1).long()        # (B,)
        else:
            num_atoms = torch.full((B,), Na, device=device, dtype=torch.long)
        # find nearest bucket
        buckets = torch.tensor(self._atom_buckets, device=device)
        bucket_idx = (num_atoms.unsqueeze(1) - buckets.unsqueeze(0)).abs().argmin(dim=1)
        natom_loss = F.cross_entropy(natom_logits, bucket_idx)
        ligand_stat_loss, ligand_stat_logits = self._ligand_stat_losses(pocket_pooled, ligand_stats)
        scaffold_fp_loss, scaffold_fp_logits = self._scaffold_fp_loss(pocket_pooled, scaffold_fp, scaffold_fp_mask)
        scaffold_token_logits = self.scaffold_token_head(h_clean)
        scaffold_token_target = self._scaffold_token_targets(lig_atom_type, lig_mask, ligand_stats)
        if lig_mask is not None:
            scaffold_valid = (~lig_mask).reshape(-1)
            scaffold_token_loss = F.cross_entropy(
                scaffold_token_logits.reshape(-1, 3)[scaffold_valid],
                scaffold_token_target.reshape(-1)[scaffold_valid],
            )
        else:
            scaffold_token_loss = F.cross_entropy(scaffold_token_logits.reshape(-1, 3), scaffold_token_target.reshape(-1))

        # Rebalanced weights: after coord centralisation coord_loss is ~1-10,
        # so type/bond/natom now receive proper gradient signal.
        loss = coord_loss + 1.0 * type_loss + 0.5 * bond_loss + 0.5 * natom_loss + 0.8 * ligand_stat_loss + 0.4 * scaffold_token_loss + 0.3 * scaffold_fp_loss

        return dict(
            loss       = loss,
            coord_loss = coord_loss,
            type_loss  = type_loss,
            bond_loss  = bond_loss,
            natom_loss = natom_loss,
            ligand_stat_loss = ligand_stat_loss,
            scaffold_token_loss = scaffold_token_loss,
            scaffold_fp_loss = scaffold_fp_loss,
            ligand_stat_logits = ligand_stat_logits,
            scaffold_token_logits = scaffold_token_logits,
            scaffold_fp_logits = scaffold_fp_logits,
        )

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        rec_feats:   torch.Tensor,
        rec_centers: torch.Tensor,
        rec_mask:    Optional[torch.Tensor] = None,
        num_atoms:   Optional[int] = None,
        pocket_center: Optional[torch.Tensor] = None,   # (B, 3) anchor
        temperature: float = 1.0,
        natom_mode: str = "auto",
        direct: bool = False,
        scaffold_prior_strength: float = 0.0,
    ) -> dict:
        """
        Run full DDPM reverse to generate one ligand per batch entry.

        Parameters
        ----------
        rec_feats    : (B, Tp, K, 6)
        rec_centers  : (B, Tp, 3)
        rec_mask     : (B, Tp)  True=pad
        num_atoms    : override number of atoms (else predict from pocket)
        pocket_center: (B, 3)  ligand will be initialised near this point
        temperature  : noise scale during sampling (1.0 = standard)
        natom_mode   : atom-count selection mode: auto/top1/topk_sample/expected
        scaffold_prior_strength: soft scaffold-fingerprint prior strength (0 disables)

        Returns
        -------
        dict with:
          positions  : (B, Na, 3)   generated atom positions
          atom_types : (B, Na)      element indices (ELEMENT_LIST)
          bond_logits: (B, Na, Na, 5) raw bond-type logits
          h_final    : (B, Na, D)   atom hidden states (for downstream use)
        """
        B      = rec_feats.shape[0]
        device = rec_feats.device

        # encode pocket
        pocket_tokens, pocket_pooled = self._encode_pocket(
            rec_feats, rec_centers, rec_mask)

        ligand_stat_logits = [head(pocket_pooled) for head in self.ligand_stat_heads]

        graph_counts = F.softplus(self.graph_count_head(pocket_pooled))
        scaffold_fp_logits = self.scaffold_fp_head(pocket_pooled)
        scaffold_fp_prob = scaffold_fp_logits.sigmoid()
        scaffold_fp_on_bits = (scaffold_fp_prob > 0.5).float().sum(dim=-1)
        scaffold_fp_mass = scaffold_fp_prob.sum(dim=-1)
        scaffold_fp_density = (scaffold_fp_mass / float(SCAFFOLD_FP_BITS)).clamp(0.0, 1.0)
        scaffold_complexity = scaffold_fp_density.clamp(0.0, 1.0)
        ring_prior_boost = scaffold_complexity
        aromatic_prior_boost = scaffold_complexity.pow(1.2)
        hetero_prior_boost = (0.5 * scaffold_complexity + 0.5 * (scaffold_fp_on_bits / float(SCAFFOLD_FP_BITS)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        branch_prior_boost = scaffold_complexity.sqrt()

        prior_strength = float(max(0.0, scaffold_prior_strength))
        if prior_strength > 0.0:
            adjusted_logits = []
            for stat_idx, logits in enumerate(ligand_stat_logits):
                bins = logits.shape[-1]
                axis = torch.linspace(0.0, 1.0, bins, device=device, dtype=logits.dtype).view(1, bins)
                if stat_idx == 0:
                    simple_bias = (0.35 - scaffold_complexity).unsqueeze(-1) * (1.0 - axis)
                    complex_bias = (scaffold_complexity - 0.25).unsqueeze(-1) * axis
                    bias = simple_bias + complex_bias
                    if bins > 4:
                        bias[:, min(4, bins - 1):] = bias[:, min(4, bins - 1):] + aromatic_prior_boost.unsqueeze(-1)
                elif stat_idx == 1:
                    bias = ring_prior_boost.unsqueeze(-1) * axis
                elif stat_idx == 2:
                    bias = aromatic_prior_boost.unsqueeze(-1) * axis
                elif stat_idx == 3:
                    bias = hetero_prior_boost.unsqueeze(-1) * axis
                elif stat_idx == 4:
                    bias = -scaffold_complexity.unsqueeze(-1) * axis
                elif stat_idx == 5:
                    bias = branch_prior_boost.unsqueeze(-1) * axis
                elif stat_idx == 8:
                    bias = branch_prior_boost.unsqueeze(-1) * axis
                else:
                    bias = torch.zeros_like(logits)
                adjusted_logits.append(logits + prior_strength * bias)
            ligand_stat_logits = adjusted_logits
            graph_counts = graph_counts.clone()
            graph_counts[:, 0] = graph_counts[:, 0] + prior_strength * (0.8 * ring_prior_boost + 0.4 * branch_prior_boost)
            graph_counts[:, 1] = graph_counts[:, 1] + prior_strength * (2.0 * ring_prior_boost)
            graph_counts[:, 2] = graph_counts[:, 2] + prior_strength * (3.0 * branch_prior_boost)

        ligand_stats_pred = torch.stack([logit.argmax(dim=-1) for logit in ligand_stat_logits], dim=-1)

        # predict number of atoms
        natom_logits = self.natom_head(pocket_pooled)
        prob = (natom_logits / max(float(temperature), 1e-3)).softmax(dim=-1)
        buckets = torch.tensor(self._atom_buckets, device=device, dtype=prob.dtype)
        expected = (prob[0] * buckets).sum()
        topk = torch.topk(prob[0], k=min(8, prob.shape[-1]))
        if num_atoms is None:
            mode = str(natom_mode).lower()
            if mode not in {"auto", "top1", "topk_sample", "expected"}:
                raise ValueError("natom_mode must be one of: auto, top1, topk_sample, expected")
            if mode == "auto":
                mode = "top1" if temperature <= 0.75 else "topk_sample"
            if mode == "top1":
                selected_bucket = topk.indices[0] + 1
                natom_source = "top1"
            elif mode == "expected":
                selected_bucket = torch.round(expected).long()
                natom_source = "expected"
            else:
                top_probs = topk.values / topk.values.sum().clamp(min=1e-8)
                pick = torch.multinomial(top_probs, num_samples=1).item()
                selected_bucket = topk.indices[pick] + 1
                natom_source = "topk_sample"
            num_atoms = int(selected_bucket.clamp(4, min(48, self.max_atoms)).item())
        else:
            natom_source = "override"

        Na = num_atoms

        # initialise ligand atom types to Carbon (index 0) — will be refined
        atom_type = torch.zeros(B, Na, dtype=torch.long, device=device)

        # start from Gaussian noise centred on pocket_center
        if pocket_center is not None:
            center = pocket_center.unsqueeze(1)             # (B, 1, 3)
        else:
            # use mean of valid pocket patch centres as anchor
            valid = (~rec_mask).float().unsqueeze(-1) \
                if rec_mask is not None \
                else torch.ones(B, rec_centers.shape[1], 1, device=device)
            center = (rec_centers * valid).sum(1, keepdim=True) / \
                     valid.sum(1, keepdim=True).clamp(min=1)  # (B,1,3)

        # Receptor-only direct mode: no ligand atom-type conditioning and no
        # reference topology.  It predicts local coordinates from pocket tokens.
        if direct:
            direct_out = self.direct_head(pocket_tokens, rec_mask, num_atoms=Na)
            positions_rel = self._spread_direct_positions(direct_out["positions_rel"])
            positions = positions_rel + (pocket_center.unsqueeze(1) if pocket_center is not None else center)
            type_logits = direct_out["type_logits"]
            atom_types = type_logits.argmax(dim=-1)
            return dict(
                positions=positions,
                atom_types=atom_types,
                bond_logits=direct_out["bond_logits"],
                degree_logits=direct_out["degree_logits"],
                graph_counts=graph_counts,
                ligand_stats_pred=ligand_stats_pred,
                ligand_stat_logits=ligand_stat_logits,
                scaffold_fp_logits=scaffold_fp_logits,
                scaffold_fp_on_bits=scaffold_fp_on_bits,
                scaffold_fp_mass=scaffold_fp_mass,
                scaffold_fp_density=scaffold_fp_density,
                scaffold_complexity=scaffold_complexity,
                ring_prior_boost=ring_prior_boost,
                aromatic_prior_boost=aromatic_prior_boost,
                hetero_prior_boost=hetero_prior_boost,
                branch_prior_boost=branch_prior_boost,
                scaffold_prior_strength=torch.full((B,), prior_strength, device=device),
                natom_expected=expected.detach(),
                natom_selected=torch.tensor(float(Na), device=device),
                natom_source=natom_source,
                natom_top_indices=topk.indices.detach(),
                natom_top_probs=topk.values.detach(),
                h_final=direct_out["h_final"],
            )

        # Training is performed in a ligand-centred local frame, so sampling
        # must also start in that local frame.  The absolute pocket anchor is
        # added back only once after reverse diffusion.
        x_t = temperature * torch.randn(B, Na, 3, device=device) * 5.0

        # reverse diffusion
        for step in reversed(range(self.T)):
            t_batch = torch.full((B,), step, device=device, dtype=torch.long)
            x0_pred = self.denoiser(x_t, atom_type, t_batch,
                                    pocket_tokens, rec_mask)
            x_t = self.schedule.p_sample(x_t, step, x0_pred)

        # final positions in the local ligand-centred frame
        positions_rel = x_t    # (B, Na, 3)

        if pocket_center is not None:
            positions = positions_rel + pocket_center.unsqueeze(1)  # (B, Na, 3)
        else:
            positions = positions_rel + center  # center is (B,1,3)

        # predict atom types from final hidden state
        t0 = torch.zeros(B, device=device, dtype=torch.long)
        h  = self.denoiser.pos_proj(positions_rel) + \
             self.denoiser.atom_emb(atom_type) + \
             self.denoiser.t_proj(self.denoiser.t_emb(t0)).unsqueeze(1)
        for blk in self.denoiser.blocks:
            h = blk(h, pocket_tokens, rec_mask)
        h = self.denoiser.out_norm(h)

        type_logits = self.type_head(h)            # (B, Na, E)
        atom_types  = type_logits.argmax(dim=-1)   # (B, Na)

        bond_logits = self.bond_head(h)            # (B, Na, Na, 5)

        degree_logits = self.direct_head.degree_head(h)

        return dict(
            positions   = positions,
            atom_types  = atom_types,
            bond_logits = bond_logits,
            degree_logits = degree_logits,
            graph_counts = graph_counts,
            ligand_stats_pred = ligand_stats_pred,
            ligand_stat_logits = ligand_stat_logits,
            scaffold_fp_logits = scaffold_fp_logits,
            scaffold_fp_on_bits = scaffold_fp_on_bits,
            scaffold_fp_mass = scaffold_fp_mass,
            scaffold_fp_density = scaffold_fp_density,
            scaffold_complexity = scaffold_complexity,
            ring_prior_boost = ring_prior_boost,
            aromatic_prior_boost = aromatic_prior_boost,
            hetero_prior_boost = hetero_prior_boost,
            branch_prior_boost = branch_prior_boost,
            scaffold_prior_strength = torch.full((B,), prior_strength, device=device),
            natom_expected = expected.detach(),
            natom_selected = torch.tensor(float(Na), device=device),
            natom_source = natom_source,
            natom_top_indices = topk.indices.detach(),
            natom_top_probs = topk.values.detach(),
            h_final     = h,
        )
