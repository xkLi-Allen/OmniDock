# -*- coding: utf-8 -*-
"""
All loss functions for Stage 2 and Stage 3.
Self-contained - no imports from other projects.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage-2 losses
# ---------------------------------------------------------------------------

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b : (N, K, 3). Returns mean symmetric Chamfer distance."""
    with torch.amp.autocast("cuda", enabled=False):
        a32 = a.float(); b32 = b.float()
        D   = torch.cdist(a32, b32)           # (N, K, K)
        a2b = D.min(dim=2).values
        b2a = D.min(dim=1).values
        cd  = (a2b.pow(2).mean(dim=1) + b2a.pow(2).mean(dim=1)).mean()
    return cd


def curvature_proxy(coords: torch.Tensor) -> torch.Tensor:
    """coords : (N, K, 3). Returns (N,) curvature proxy via PCA min-eigenvalue ratio."""
    with torch.amp.autocast("cuda", enabled=False):
        x  = coords.float()
        N, K, _ = x.shape
        x  = x - x.mean(dim=1, keepdim=True)
        cov = torch.einsum("nki,nkj->nij", x, x) / (max(K - 1, 1) + 1e-6)
        ev  = torch.linalg.eigvalsh(cov).clamp_min(0.0)   # (N, 3)
        kappa = ev[:, 0] / (ev.sum(dim=1) + 1e-8)
    return kappa.to(coords.dtype)


def kl_to_uniform(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """KL( softmax(logits/tau) || Uniform(C) ). logits: (*, C)."""
    C = logits.shape[-1]
    q = torch.softmax(logits / max(tau, 1e-4), dim=-1)
    return ((q.clamp_min(1e-9).log() * q).sum(dim=-1) + math.log(C)).mean()


# ---------------------------------------------------------------------------
# Stage-3 losses
# ---------------------------------------------------------------------------

def pocket_bce_loss(pocket_logits, pocket_labels, rec_mask):
    """
    pocket_logits : (B, Tr)
    pocket_labels : (B, Tr)  float32 in {0,1}
    rec_mask      : (B, Tr)  True = padding
    """
    valid = ~rec_mask
    if valid.sum() == 0:
        return pocket_logits.new_tensor(0.0)
    lv = pocket_logits[valid]
    yv = pocket_labels[valid].float()
    pf = yv.mean().clamp(1e-4, 1.0)
    pw = torch.tensor(float((1.0 - pf) / pf), device=lv.device)
    return F.binary_cross_entropy_with_logits(lv, yv, pos_weight=pw)


def binding_bce_loss(bind_logits, bind_labels):
    """bind_logits, bind_labels : (B,)."""
    return F.binary_cross_entropy_with_logits(
        bind_logits, bind_labels.float())


def affinity_loss(aff_pred, aff_gt, is_pos):
    """SmoothL1 on positive pairs only."""
    if not is_pos.any():
        return aff_pred.new_tensor(0.0)
    return F.smooth_l1_loss(aff_pred[is_pos], aff_gt[is_pos].float())


def contact_loss(rec_tokens, lig_tokens, rec_centers, lig_pos,
                 rec_mask, lig_mask, contact_thresh=6.0):
    """
    BCE on patch-atom contact matrix.
    rec_tokens  : (B, Tr, D)
    lig_tokens  : (B, Na, D)   (ligand atom embeddings)
    rec_centers : (B, Tr, 3)
    lig_pos     : (B, Na, 3)
    """
    B, D = rec_tokens.shape[0], rec_tokens.shape[-1]
    total = rec_tokens.new_tensor(0.0)
    used  = 0
    for b in range(B):
        vr = ~rec_mask[b]; vl = ~lig_mask[b]
        if vr.sum() == 0 or vl.sum() == 0:
            continue
        rt = rec_tokens[b][vr]                       # (Tr_v, D)
        lt = lig_tokens[b][vl]                       # (Na_v, D)
        rc = rec_centers[b][vr]                      # (Tr_v, 3)
        lp = lig_pos[b][vl]                          # (Na_v, 3)
        with torch.amp.autocast("cuda", enabled=False):
            Dmat = torch.cdist(rc.float(), lp.float()) # (Tr_v, Na_v)
        contact = (Dmat <= contact_thresh).float()
        scores  = (rt @ lt.t()) / math.sqrt(D)
        pf = contact.mean().clamp(1e-4, 1.0)
        pw = torch.tensor(float((1.0 - pf) / pf), device=scores.device)
        total += F.binary_cross_entropy_with_logits(
            scores, contact, pos_weight=pw)
        used += 1
    return total / max(used, 1)


def flex_loss(tokens, centers, mask, k_nbr=8):
    """
    Local smoothness: token embedding should vary smoothly over surface.
    tokens  : (B, T, D)
    centers : (B, T, 3)
    mask    : (B, T) True=pad
    """
    total = tokens.new_tensor(0.0)
    used  = 0
    for b in range(tokens.shape[0]):
        v = ~mask[b]
        h = tokens[b][v]; c = centers[b][v]
        Tv = h.shape[0]
        if Tv <= 1:
            continue
        with torch.amp.autocast("cuda", enabled=False):
            Dmat = torch.cdist(c.float(), c.float())
        k = min(k_nbr + 1, Tv)
        _, nn_idx = torch.topk(Dmat, k=k, dim=-1, largest=False)
        nn_idx = nn_idx[:, 1:]                       # exclude self
        diff = h.unsqueeze(1) - h[nn_idx]            # (Tv, k-1, D)
        total += diff.pow(2).mean()
        used  += 1
    return total / max(used, 1)
