# -*- coding: utf-8 -*-
"""
Stage 4 training: train TorsionGenerator to predict ligand backbone torsions
and secondary structure from receptor surface condition.

This gives Stage4 a real learned generator instead of random torsion output.
"""

import os
import sys
import random
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage3_dataset import SkempiDockingDataset, stage3_collate_fn
from stage4_model import TorsionGenerator, TOPOLOGY_FAMILIES
from geometry_utils import (
    sincos_to_torsion, canonicalize_torsion_sincos, compute_dihedral_torch,
    BOND_C_O, ANGLE_CA_C_O,
)
from stage4_generate import load_compatible_state_dict, generate_backbone, write_pdb, validate_backbone


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_torsion_sincos(x: torch.Tensor) -> torch.Tensor:
    """Normalize each torsion's sin/cos pair independently.

    Representation is [sin_phi, sin_psi, sin_omega, cos_phi, cos_psi, cos_omega].
    """
    return canonicalize_torsion_sincos(x, assume_legacy_omega=False)


def canonicalize_legacy_target_torsion(x: torch.Tensor) -> torch.Tensor:
    return canonicalize_torsion_sincos(x, assume_legacy_omega=True)


def angular_diff_deg(a: torch.Tensor, b: float) -> torch.Tensor:
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


def omega_trans_loss(torsion_sincos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    torsion = sincos_to_torsion(normalize_torsion_sincos(torsion_sincos))
    omega = torsion[..., 2]
    if mask.any():
        return (angular_diff_deg(omega[mask], 180.0) ** 2).mean() / 1000.0
    return torsion_sincos.new_tensor(0.0)


def ss_consistency_loss(torsion_sincos: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    torsion = sincos_to_torsion(normalize_torsion_sincos(torsion_sincos))
    phi = torsion[..., 0]
    psi = torsion[..., 1]
    helix = mask & (target_ss == 0)
    beta = mask & (target_ss == 1)
    losses = []
    if helix.any():
        losses.append((angular_diff_deg(phi[helix], -57.0).pow(2) + angular_diff_deg(psi[helix], -47.0).pow(2)).mean() / 1000.0)
    if beta.any():
        losses.append((angular_diff_deg(phi[beta], -135.0).pow(2) + angular_diff_deg(psi[beta], 135.0).pow(2)).mean() / 1000.0)
    if not losses:
        return torsion_sincos.new_tensor(0.0)
    return torch.stack(losses).mean()


def ss_smoothness_loss(ss_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if ss_logits.shape[1] < 2:
        return ss_logits.new_tensor(0.0)
    valid = mask[:, 1:] & mask[:, :-1]
    if not valid.any():
        return ss_logits.new_tensor(0.0)
    probs = F.softmax(ss_logits.float(), dim=-1)
    diff = probs[:, 1:] - probs[:, :-1]
    return diff[valid].pow(2).sum(dim=-1).mean().to(ss_logits.dtype)


def ss_ce_class_weights(target_ss: torch.Tensor, mask: torch.Tensor, args, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if args.ss_class_weight == "none":
        return None
    if args.ss_class_weight == "manual":
        return torch.tensor([args.ss_weight_helix, args.ss_weight_beta, args.ss_weight_coil], dtype=dtype, device=device)
    if not mask.any():
        return torch.ones(3, dtype=dtype, device=device)
    labels = target_ss[mask]
    counts = torch.stack([(labels == i).float().sum() for i in range(3)]).to(device=device, dtype=dtype)
    freq = counts / counts.sum().clamp(min=1.0)
    weights = 1.0 / freq.clamp(min=0.05).sqrt()
    weights = weights / weights.mean().clamp(min=1e-6)
    return weights.clamp(min=0.5, max=3.0)


def _segment_interior_weights(target_ss: torch.Tensor, mask: torch.Tensor, label: int,
                              edge_weight: float = 0.5, interior_weight: float = 1.0) -> torch.Tensor:
    weights = torch.zeros_like(target_ss, dtype=torch.float32)
    B, L = target_ss.shape
    for b in range(B):
        start = 0
        while start < L:
            is_seg = bool(mask[b, start].item()) and int(target_ss[b, start].item()) == label
            end = start + 1
            while end < L and bool(mask[b, end].item()) and int(target_ss[b, end].item()) == label:
                end += 1
            if is_seg:
                seg_len = end - start
                weights[b, start:end] = interior_weight
                if seg_len >= 2:
                    weights[b, start] = edge_weight
                    weights[b, end - 1] = edge_weight
            start = end if is_seg else start + 1
    return weights.to(device=target_ss.device)


def target_ss_torsion_loss(torsion_sincos: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torsion = sincos_to_torsion(normalize_torsion_sincos(torsion_sincos))
    phi = torsion[..., 0]
    psi = torsion[..., 1]
    helix_mask = mask & (target_ss == 0)
    beta_mask = mask & (target_ss == 1)
    helix_w = _segment_interior_weights(target_ss, helix_mask, 0).to(dtype=phi.dtype)
    beta_w = _segment_interior_weights(target_ss, beta_mask, 1).to(dtype=phi.dtype)

    l_helix = torsion_sincos.new_tensor(0.0)
    if helix_mask.any():
        h = (angular_diff_deg(phi, -60.0).pow(2) + angular_diff_deg(psi, -45.0).pow(2)) / 1000.0
        l_helix = (h * helix_w).sum() / helix_w.sum().clamp(min=1.0)

    l_beta = torsion_sincos.new_tensor(0.0)
    if beta_mask.any():
        e = (angular_diff_deg(phi, -135.0).pow(2) + angular_diff_deg(psi, 135.0).pow(2)) / 1000.0
        l_beta = (e * beta_w).sum() / beta_w.sum().clamp(min=1.0)

    return l_helix + l_beta, l_helix, l_beta


def _place_carbonyl_oxygen(backbone: torch.Tensor) -> torch.Tensor:
    """Approximate backbone carbonyl O positions from N-CA-C geometry."""
    n = backbone[:, :, 0, :]
    ca = backbone[:, :, 1, :]
    c = backbone[:, :, 2, :]
    c_ca = ca - c
    c_n = n - c
    e1 = c_ca / c_ca.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c_n_proj = c_n - (c_n * e1).sum(dim=-1, keepdim=True) * e1
    e2 = c_n_proj / c_n_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    theta = ANGLE_CA_C_O * (math.pi / 180.0)
    direction = math.cos(theta) * e1 - math.sin(theta) * e2
    return c + BOND_C_O * direction


def backbone_phi_psi(backbone: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = backbone if backbone.dim() == 4 else backbone.unsqueeze(0)
    n = backbone[:, :, 0, :]
    ca = backbone[:, :, 1, :]
    c = backbone[:, :, 2, :]
    phi = compute_dihedral_torch(c[:, :-1], n[:, 1:], ca[:, 1:], c[:, 1:])
    psi = compute_dihedral_torch(n[:, :-1], ca[:, :-1], c[:, :-1], n[:, 1:])
    return phi, psi


def dssp_like_ss_from_backbone(backbone: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast diagnostic DSSP proxy from generated geometry: 0=H, 1=E, 2=C.

    This proxy is intentionally diagnostic. Training supervision should come
    from target DSSP labels driving torsion/segment geometry, not from this
    generated-geometry label pulling the SS head toward coil.
    """
    phi, psi = backbone_phi_psi(backbone)
    valid = mask[:, 1:] & mask[:, :-1]
    d_helix = angular_diff_deg(phi, -60.0).pow(2) + angular_diff_deg(psi, -45.0).pow(2)
    d_beta = angular_diff_deg(phi, -135.0).pow(2) + angular_diff_deg(psi, 135.0).pow(2)
    ss_mid = torch.full(phi.shape, 2, dtype=torch.long, device=backbone.device)

    if backbone.shape[1] >= 5:
        ca = backbone[:, :, 1, :]
        ca_i4 = (ca[:, 4:] - ca[:, :-4]).norm(dim=-1)
        ca_i2 = (ca[:, 2:] - ca[:, :-2]).norm(dim=-1)
        helix_i4 = (ca_i4 >= 4.8) & (ca_i4 <= 6.8) & mask[:, :-4] & mask[:, 4:]
        helix_i2 = (ca_i2 >= 5.0) & (ca_i2 <= 5.8) & mask[:, :-2] & mask[:, 2:]
        helix_geom_res = torch.zeros_like(mask, dtype=torch.bool)
        helix_geom_res[:, :-4] |= helix_i4
        helix_geom_res[:, 1:-3] |= helix_i4
        helix_geom_res[:, 2:-2] |= helix_i4
        helix_geom_res[:, 3:-1] |= helix_i4
        helix_geom_res[:, 4:] |= helix_i4
        helix_geom_res[:, :-2] |= helix_i2
        helix_geom_res[:, 1:-1] |= helix_i2
        helix_geom_res[:, 2:] |= helix_i2
        ss_mid = torch.where((d_helix < 3600.0) & helix_geom_res[:, 1:], torch.zeros_like(ss_mid), ss_mid)
    else:
        ss_mid = torch.where(d_helix < 2500.0, torch.zeros_like(ss_mid), ss_mid)

    beta_geom_res = torch.zeros_like(mask, dtype=torch.bool)
    if backbone.shape[1] >= 4:
        ca = backbone[:, :, 1, :]
        ca_i2 = (ca[:, 2:] - ca[:, :-2]).norm(dim=-1)
        ca_i3 = (ca[:, 3:] - ca[:, :-3]).norm(dim=-1)
        ext_i2 = (ca_i2 >= 6.0) & (ca_i2 <= 7.6) & mask[:, :-2] & mask[:, 2:]
        ext_i3 = (ca_i3 >= 8.5) & (ca_i3 <= 11.5) & mask[:, :-3] & mask[:, 3:]
        beta_geom_res[:, :-2] |= ext_i2
        beta_geom_res[:, 1:-1] |= ext_i2
        beta_geom_res[:, 2:] |= ext_i2
        beta_geom_res[:, :-3] |= ext_i3
        beta_geom_res[:, 1:-2] |= ext_i3
        beta_geom_res[:, 2:-1] |= ext_i3
        beta_geom_res[:, 3:] |= ext_i3
    ss_mid = torch.where((ss_mid == 2) & (d_beta < 3600.0) & beta_geom_res[:, 1:], torch.ones_like(ss_mid), ss_mid)

    return ss_mid, valid


def dssp_proxy_metrics(pred_bb: torch.Tensor, pred_ss_logits: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor) -> dict:
    proxy_ss, proxy_valid = dssp_like_ss_from_backbone(pred_bb, mask)
    if not proxy_valid.any():
        return {"dssp_proxy_acc": 0.0, "dssp_proxy_target_acc": 0.0, "dssp_proxy_helix_frac": 0.0, "dssp_proxy_beta_frac": 0.0, "dssp_proxy_coil_frac": 0.0}
    pred_label = pred_ss_logits.detach().argmax(dim=-1)[:, 1:]
    target_mid = target_ss.detach()[:, 1:]
    proxy = proxy_ss.detach()[proxy_valid]
    pred = pred_label[proxy_valid]
    target = target_mid[proxy_valid]
    return {
        "dssp_proxy_acc": float((pred == proxy).float().mean().item()),
        "dssp_proxy_target_acc": float((target == proxy).float().mean().item()),
        "dssp_proxy_helix_frac": float((proxy == 0).float().mean().item()),
        "dssp_proxy_beta_frac": float((proxy == 1).float().mean().item()),
        "dssp_proxy_coil_frac": float((proxy == 2).float().mean().item()),
    }


def ss_geometry_agreement_loss(pred_bb: torch.Tensor, pred_ss_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    proxy_ss, proxy_valid = dssp_like_ss_from_backbone(pred_bb, mask)
    if not proxy_valid.any():
        return pred_bb.new_tensor(0.0)
    return F.cross_entropy(pred_ss_logits[:, 1:][proxy_valid], proxy_ss[proxy_valid])


def helix_hbond_loss(pred_bb: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor,
                     target_dist: float = 2.9, cutoff: float = 3.5) -> torch.Tensor:
    """Encourage DSSP-defining alpha-helix C=O(i) ... H-N(i+4) geometry."""
    if pred_bb.shape[1] < 5:
        return pred_bb.new_tensor(0.0)
    o = _place_carbonyl_oxygen(pred_bb)
    n = pred_bb[:, :, 0, :]
    dist = (o[:, :-4] - n[:, 4:]).norm(dim=-1)
    helix_window = (
        (target_ss[:, :-4] == 0) & (target_ss[:, 1:-3] == 0) &
        (target_ss[:, 2:-2] == 0) & (target_ss[:, 3:-1] == 0) & (target_ss[:, 4:] == 0)
    )
    valid = mask[:, :-4] & mask[:, 4:] & helix_window
    if not valid.any():
        return pred_bb.new_tensor(0.0)
    return (F.relu(dist[valid] - cutoff).pow(2) + 0.1 * (dist[valid] - target_dist).pow(2)).mean()


def helix_i4_ca_loss(pred_bb: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor,
                     min_dist: float = 5.4, max_dist: float = 6.2) -> torch.Tensor:
    if pred_bb.shape[1] < 5:
        return pred_bb.new_tensor(0.0)
    ca = pred_bb[:, :, 1, :]
    dist = (ca[:, 4:] - ca[:, :-4]).norm(dim=-1)
    helix_window = (
        (target_ss[:, :-4] == 0) & (target_ss[:, 1:-3] == 0) &
        (target_ss[:, 2:-2] == 0) & (target_ss[:, 3:-1] == 0) & (target_ss[:, 4:] == 0)
    )
    valid = mask[:, :-4] & mask[:, 4:] & helix_window
    if not valid.any():
        return pred_bb.new_tensor(0.0)
    center = 0.5 * (min_dist + max_dist)
    return (F.relu(min_dist - dist[valid]).pow(2) + F.relu(dist[valid] - max_dist).pow(2) + 0.05 * (dist[valid] - center).pow(2)).mean()


def beta_extended_geometry_loss(pred_bb: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    beta = mask & (target_ss == 1)
    if pred_bb.shape[1] < 4 or not beta.any():
        return pred_bb.new_tensor(0.0)
    ca = pred_bb[:, :, 1, :]
    losses = []
    if pred_bb.shape[1] > 2:
        valid2 = beta[:, :-2] & beta[:, 1:-1] & beta[:, 2:]
        if valid2.any():
            d2 = (ca[:, 2:] - ca[:, :-2]).norm(dim=-1)
            losses.append((F.relu(6.0 - d2[valid2]).pow(2) + F.relu(d2[valid2] - 7.6).pow(2)).mean())
    if pred_bb.shape[1] > 3:
        valid3 = beta[:, :-3] & beta[:, 1:-2] & beta[:, 2:-1] & beta[:, 3:]
        if valid3.any():
            d3 = (ca[:, 3:] - ca[:, :-3]).norm(dim=-1)
            losses.append((F.relu(8.5 - d3[valid3]).pow(2) + F.relu(d3[valid3] - 11.5).pow(2)).mean())
    if not losses:
        return pred_bb.new_tensor(0.0)
    return torch.stack(losses).mean()


def beta_hbond_geometry_loss(pred_bb: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor,
                             target_dist: float = 2.9, cutoff: float = 3.8) -> torch.Tensor:
    """Encourage at least one plausible long-range backbone H-bond contact for beta residues."""
    beta = mask & (target_ss == 1)
    if pred_bb.shape[1] < 6 or not beta.any():
        return pred_bb.new_tensor(0.0)
    o = _place_carbonyl_oxygen(pred_bb)
    n = pred_bb[:, :, 0, :]
    d = torch.cdist(o.float(), n.float()).to(pred_bb.dtype)
    L = pred_bb.shape[1]
    idx = torch.arange(L, device=pred_bb.device)
    sep = (idx[None, :] - idx[:, None]).abs().unsqueeze(0)
    pair_valid = beta.unsqueeze(2) & beta.unsqueeze(1) & (sep >= 3)
    if not pair_valid.any():
        return pred_bb.new_tensor(0.0)
    d = d.masked_fill(~pair_valid, 1e6)
    nearest = d.min(dim=-1).values
    valid_beta = beta & torch.isfinite(nearest) & (nearest < 1e5)
    if not valid_beta.any():
        return pred_bb.new_tensor(0.0)
    return (F.relu(nearest[valid_beta] - cutoff).pow(2) + 0.05 * (nearest[valid_beta] - target_dist).pow(2)).mean()


def _extract_segments_1d(labels: torch.Tensor, valid: torch.Tensor, label: int, min_len: int) -> list[tuple[int, int, int]]:
    segments = []
    L = labels.shape[0]
    i = 0
    while i < L:
        if bool(valid[i].item()) and int(labels[i].item()) == label:
            j = i + 1
            while j < L and bool(valid[j].item()) and int(labels[j].item()) == label:
                j += 1
            if j - i >= min_len:
                segments.append((i, j - 1, j - i))
            i = j
        else:
            i += 1
    return segments


def _paired_indices(seg_a: tuple[int, int, int], seg_b: tuple[int, int, int], orientation: str,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(seg_a[2], seg_b[2])
    ia = torch.arange(seg_a[0], seg_a[0] + length, device=device, dtype=torch.long)
    if orientation == "parallel":
        ib = torch.arange(seg_b[0], seg_b[0] + length, device=device, dtype=torch.long)
    else:
        ib = torch.arange(seg_b[1], seg_b[1] - length, -1, device=device, dtype=torch.long)
    return ia, ib


def _choose_beta_pair_candidates(target_ss_b: torch.Tensor, mask_b: torch.Tensor, target_ca_b: torch.Tensor | None,
                                 min_len: int, mode: str, orientation_mode: str,
                                 max_pairs: int = 4) -> list[tuple[tuple[int, int, int], tuple[int, int, int], str]]:
    segments = _extract_segments_1d(target_ss_b, mask_b, 1, min_len)
    candidates = []
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1:]:
            if seg_b[0] - seg_a[1] < 4:
                continue
            orientations = ["parallel", "antiparallel"] if orientation_mode == "both" else [orientation_mode]
            for orient in orientations:
                score = float(seg_b[0] - seg_a[1])
                if mode == "native_ca" and target_ca_b is not None:
                    ia, ib = _paired_indices(seg_a, seg_b, orient, target_ca_b.device)
                    d = (target_ca_b[ia] - target_ca_b[ib]).norm(dim=-1)
                    in_sheet = F.relu(4.5 - d).pow(2) + F.relu(d - 6.5).pow(2)
                    score = float((in_sheet.mean() + 0.05 * d.mean()).detach().item())
                candidates.append((score, seg_a, seg_b, orient))
    candidates.sort(key=lambda x: x[0])
    return [(a, b, o) for _, a, b, o in candidates[:max_pairs]]


def beta_sheet_pairing_losses(pred_bb: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor,
                              target_ca: torch.Tensor | None, min_segment_len: int,
                              pair_mode: str, orientation_mode: str,
                              max_pair_candidates: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ca = pred_bb[:, :, 1, :]
    o = _place_carbonyl_oxygen(pred_bb)
    n = pred_bb[:, :, 0, :]
    ca_losses = []
    hbond_losses = []
    dir_losses = []
    pair_count = 0
    paired_masks = []
    for b in range(pred_bb.shape[0]):
        native_ca_b = target_ca[b] if target_ca is not None else None
        paired_mask_b = torch.zeros((pred_bb.shape[1], pred_bb.shape[1]), dtype=torch.bool, device=pred_bb.device)
        pairs = _choose_beta_pair_candidates(
            target_ss[b], mask[b], native_ca_b, min_segment_len, pair_mode, orientation_mode,
            max_pairs=max_pair_candidates
        )
        for seg_a, seg_b, orient in pairs:
            ia, ib = _paired_indices(seg_a, seg_b, orient, pred_bb.device)
            if ia.numel() < 2:
                continue
            paired_mask_b[ia, ib] = True
            paired_mask_b[ib, ia] = True
            d_ca = (ca[b, ia] - ca[b, ib]).norm(dim=-1)
            ca_losses.append((F.relu(4.5 - d_ca).pow(2) + F.relu(d_ca - 6.5).pow(2) + 0.02 * (d_ca - 5.2).pow(2)).mean())

            d_on_ab = (o[b, ia] - n[b, ib]).norm(dim=-1)
            d_on_ba = (o[b, ib] - n[b, ia]).norm(dim=-1)
            d_on = torch.minimum(d_on_ab, d_on_ba)
            hbond_losses.append((F.relu(d_on - 3.5).pow(2) + 0.05 * (d_on - 2.9).pow(2)).mean())

            va = ca[b, ia[1:]] - ca[b, ia[:-1]]
            vb = ca[b, ib[1:]] - ca[b, ib[:-1]]
            cos = F.cosine_similarity(va, vb, dim=-1)
            sign = 1.0 if orient == "parallel" else -1.0
            dir_losses.append(F.relu(0.2 - sign * cos).pow(2).mean())
            pair_count += 1
        paired_masks.append(paired_mask_b)

    zero = pred_bb.new_tensor(0.0)
    l_ca = torch.stack(ca_losses).mean() if ca_losses else zero
    l_hbond = torch.stack(hbond_losses).mean() if hbond_losses else zero
    l_dir = torch.stack(dir_losses).mean() if dir_losses else zero
    paired_mask = torch.stack(paired_masks, dim=0) if paired_masks else torch.zeros((pred_bb.shape[0], pred_bb.shape[1], pred_bb.shape[1]), dtype=torch.bool, device=pred_bb.device)
    return l_ca, l_hbond, l_dir, pred_bb.new_tensor(float(pair_count)), paired_mask


def nonpair_repulsion_loss(pred_ca: torch.Tensor, mask: torch.Tensor, paired_mask: torch.Tensor | None,
                           min_dist: float = 4.0) -> torch.Tensor:
    L = pred_ca.shape[1]
    if L < 4:
        return pred_ca.new_tensor(0.0)
    valid = mask.unsqueeze(1) & mask.unsqueeze(2)
    idx = torch.arange(L, device=pred_ca.device)
    sep = (idx[None, :] - idx[:, None]).abs().unsqueeze(0)
    valid = valid & (sep > 2)
    if paired_mask is not None:
        valid = valid & (~paired_mask)
    triu = torch.triu(torch.ones((L, L), dtype=torch.bool, device=pred_ca.device), diagonal=1).unsqueeze(0)
    valid = valid & triu
    if not valid.any():
        return pred_ca.new_tensor(0.0)
    d = torch.cdist(pred_ca.float(), pred_ca.float()).to(pred_ca.dtype)
    return F.relu(min_dist - d[valid]).pow(2).mean()


def rg_guard_loss(pred_rg: torch.Tensor, true_rg: torch.Tensor, min_ratio: float, max_ratio: float) -> torch.Tensor:
    ratio = pred_rg / true_rg.clamp(min=1e-6)
    return (F.relu(min_ratio - ratio).pow(2) + F.relu(ratio - max_ratio).pow(2)).mean()


def ss_metrics(pred_ss: torch.Tensor, target_ss: torch.Tensor, mask: torch.Tensor) -> dict:
    if not mask.any():
        return {k: 0.0 for k in [
            'ss_acc', 'ss_helix_precision', 'ss_helix_recall', 'ss_beta_precision', 'ss_beta_recall',
            'ss_coil_precision', 'ss_coil_recall', 'ss_pred_helix_frac', 'ss_pred_beta_frac',
            'ss_pred_coil_frac', 'ss_target_helix_frac', 'ss_target_beta_frac', 'ss_target_coil_frac',
            'ss_helix_f1', 'ss_beta_f1', 'ss_coil_f1', 'ss_macro_f1', 'ss_balanced_acc']}
    pred = pred_ss.detach().argmax(dim=-1)[mask]
    target = target_ss.detach()[mask]
    out = {'ss_acc': float((pred == target).float().mean().item())}
    recalls = []
    f1s = []
    names = [(0, 'helix'), (1, 'beta'), (2, 'coil')]
    for label, name in names:
        p = pred == label
        t = target == label
        tp = (p & t).float().sum()
        precision = tp / p.float().sum().clamp(min=1.0)
        recall = tp / t.float().sum().clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-8)
        out[f'ss_{name}_precision'] = float(precision.item())
        out[f'ss_{name}_recall'] = float(recall.item())
        out[f'ss_{name}_f1'] = float(f1.item())
        out[f'ss_pred_{name}_frac'] = float(p.float().mean().item())
        out[f'ss_target_{name}_frac'] = float(t.float().mean().item())
        recalls.append(recall)
        f1s.append(f1)
    out['ss_balanced_acc'] = float(torch.stack(recalls).mean().item())
    out['ss_macro_f1'] = float(torch.stack(f1s).mean().item())
    return out


def masked_rg(ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.float().unsqueeze(-1)
    n = w.sum(dim=1).clamp(min=1.0)
    center = (ca * w).sum(dim=1, keepdim=True) / n.unsqueeze(-1)
    return torch.sqrt((((ca - center) ** 2).sum(dim=-1) * mask.float()).sum(dim=1) / n.squeeze(-1).clamp(min=1.0) + 1e-8)


def pairwise_distogram_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(1) & mask.unsqueeze(2)
    L = mask.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=mask.device).unsqueeze(0)
    valid = valid & (~eye)
    if not valid.any():
        return pred_ca.new_tensor(0.0)
    with torch.amp.autocast("cuda", enabled=False):
        dp = torch.cdist(pred_ca.float(), pred_ca.float())
        dt = torch.cdist(true_ca.float(), true_ca.float())
        return F.smooth_l1_loss(dp[valid], dt[valid])


def local_ca_distance_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor, mask: torch.Tensor,
                           offsets=(2, 3, 4, 5)) -> torch.Tensor:
    losses = []
    for offset in offsets:
        if pred_ca.shape[1] <= offset:
            continue
        valid = mask[:, :-offset] & mask[:, offset:]
        if not valid.any():
            continue
        dp = (pred_ca[:, offset:] - pred_ca[:, :-offset]).norm(dim=-1)
        dt = (true_ca[:, offset:] - true_ca[:, :-offset]).norm(dim=-1)
        losses.append(F.smooth_l1_loss(dp[valid], dt[valid]))
    if not losses:
        return pred_ca.new_tensor(0.0)
    return torch.stack(losses).mean()


def kabsch_align(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    aligned = []
    for b in range(pred.shape[0]):
        m = mask[b]
        if m.sum() < 3:
            aligned.append(pred[b])
            continue
        p = pred[b, m]
        q = target[b, m]
        pc = p.mean(dim=0, keepdim=True)
        qc = q.mean(dim=0, keepdim=True)
        p0 = p - pc
        q0 = q - qc
        h = p0.transpose(0, 1) @ q0
        u, _, vh = torch.linalg.svd(h.float(), full_matrices=False)
        r = vh.transpose(-2, -1) @ u.transpose(-2, -1)
        if torch.det(r.float()).item() < 0:
            vh = vh.clone()
            vh[-1, :] *= -1
            r = vh.transpose(-2, -1) @ u.transpose(-2, -1)
        aligned_b = (pred[b].float() - pc.float()) @ r.float() + qc.float()
        aligned.append(aligned_b.to(pred.dtype))
    return torch.stack(aligned, dim=0)


def ca_rmsd_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    aligned = kabsch_align(pred_ca, true_ca, mask)
    if not mask.any():
        return pred_ca.new_tensor(0.0)
    diff = ((aligned - true_ca) ** 2).sum(dim=-1)
    return torch.sqrt((diff * mask.float()).sum() / mask.float().sum().clamp(min=1.0) + 1e-8)


def kabsch_aligned_coord_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return pred_ca.new_tensor(0.0)
    aligned = kabsch_align(pred_ca, true_ca, mask)
    return F.smooth_l1_loss(aligned[mask], true_ca[mask])


def ca_self_clash_loss(pred_ca: torch.Tensor, mask: torch.Tensor, min_dist: float = 2.8) -> torch.Tensor:
    losses = []
    with torch.amp.autocast("cuda", enabled=False):
        for b in range(pred_ca.shape[0]):
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() < 4:
                continue
            ca = pred_ca[b, valid_idx].float()
            d = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0))[0]
            sep = (valid_idx[:, None] - valid_idx[None, :]).abs()
            pair_mask = (sep > 2) & torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
            if pair_mask.any():
                losses.append(F.relu(min_dist - d[pair_mask]).pow(2).mean())
    if not losses:
        return pred_ca.new_tensor(0.0)
    return torch.stack(losses).mean().to(pred_ca.dtype)


def interface_local_shape_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor, rec_centers: torch.Tensor,
                               rec_pad: torch.Tensor, lig_mask: torch.Tensor, cutoff: float) -> torch.Tensor:
    losses = []
    valid_rec = ~rec_pad
    with torch.amp.autocast("cuda", enabled=False):
        for b in range(pred_ca.shape[0]):
            if lig_mask[b].sum() < 3 or valid_rec[b].sum() == 0:
                continue
            rec = rec_centers[b, valid_rec[b]].float()
            true = true_ca[b].float()
            d_true = torch.cdist(true.unsqueeze(0), rec.unsqueeze(0))[0].min(dim=-1).values
            iface = lig_mask[b] & (d_true <= cutoff)
            if iface.sum() < 3:
                iface = lig_mask[b]
            losses.append(pairwise_distogram_loss(pred_ca[b:b+1].float(), true_ca[b:b+1].float(), iface.unsqueeze(0)))
            losses.append(kabsch_aligned_coord_loss(pred_ca[b:b+1].float(), true_ca[b:b+1].float(), iface.unsqueeze(0)))
    if not losses:
        return pred_ca.new_tensor(0.0)
    return torch.stack(losses).mean().to(pred_ca.dtype)


def contact_profile_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor, rec_centers: torch.Tensor, rec_pad: torch.Tensor, lig_mask: torch.Tensor, cutoff: float) -> torch.Tensor:
    aligned = kabsch_align(pred_ca, true_ca, lig_mask)
    valid_rec = ~rec_pad
    losses = []
    with torch.amp.autocast("cuda", enabled=False):
        for b in range(pred_ca.shape[0]):
            if lig_mask[b].sum() == 0 or valid_rec[b].sum() == 0:
                continue
            rec = rec_centers[b, valid_rec[b]].float()
            dp = torch.cdist(aligned[b:b + 1].float(), rec.unsqueeze(0))[0].min(dim=-1).values
            dt = torch.cdist(true_ca[b:b + 1].float(), rec.unsqueeze(0))[0].min(dim=-1).values
            m = lig_mask[b]
            cp = torch.sigmoid((cutoff - dp[m]) / 1.5)
            ct = torch.sigmoid((cutoff - dt[m]) / 1.5)
            losses.append(F.mse_loss(cp, ct))
    if not losses:
        return pred_ca.new_tensor(0.0)
    return torch.stack(losses).mean().to(pred_ca.dtype)


def infer_topology_targets(ss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    targets = []
    for b in range(ss.shape[0]):
        valid = mask[b]
        vals = ss[b, valid]
        if vals.numel() == 0:
            targets.append(5)
            continue
        h_frac = (vals == 0).float().mean().item()
        e_frac = (vals == 1).float().mean().item()
        h_segments = 0
        prev = False
        for x in (vals == 0).tolist():
            if x and not prev:
                h_segments += 1
            prev = bool(x)
        if e_frac > 0.35:
            targets.append(4)
        elif h_frac > 0.75:
            targets.append(0)
        elif h_segments >= 4:
            targets.append(3)
        elif h_segments >= 3:
            targets.append(2)
        elif h_segments >= 2:
            targets.append(1)
        else:
            targets.append(5)
    return torch.tensor(targets, dtype=torch.long, device=ss.device)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, global_step=0):
    model.train()
    total_loss = 0.0
    metric_sums = {}
    n_steps = 0
    pbar = tqdm(loader, desc=f"Stage4 Epoch {epoch}")

    for it, batch in enumerate(pbar):
        if args.max_steps_per_epoch > 0 and it >= args.max_steps_per_epoch:
            break

        rec_feats = batch["rec_feats"].to(device, non_blocking=True)
        rec_centers = batch["rec_centers"].to(device, non_blocking=True)
        rec_pad = batch["rec_pad"].to(device, non_blocking=True)
        target_torsion = batch["lig_torsion_target"].to(device, non_blocking=True)
        target_ss = batch["lig_ss_target"].to(device, non_blocking=True)
        ss_valid = batch.get("lig_ss_valid", batch["lig_bb_mask"]).to(device, non_blocking=True)
        target_ca = batch["lig_ca_pos"].to(device, non_blocking=True)
        lig_mask = batch["lig_bb_mask"].to(device, non_blocking=True)
        torsion_valid = batch.get("lig_torsion_valid", lig_mask).to(device, non_blocking=True)
        ss_sources = batch.get("ss_source_used", [])

        L = target_torsion.shape[1]
        if L > args.max_residues:
            target_torsion = target_torsion[:, :args.max_residues]
            target_ss = target_ss[:, :args.max_residues]
            ss_valid = ss_valid[:, :args.max_residues]
            target_ca = target_ca[:, :args.max_residues]
            lig_mask = lig_mask[:, :args.max_residues]
            torsion_valid = torsion_valid[:, :args.max_residues]
            L = args.max_residues

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            pred_torsion, pred_ss, pocket_pred, hotspot_logits, topology_logits = model(
                rec_feats, rec_centers, rec_pad, num_residues=L
            )
            pred_torsion = normalize_torsion_sincos(pred_torsion)
            target_torsion = canonicalize_legacy_target_torsion(target_torsion)

            torsion_mask = lig_mask & torsion_valid
            if torsion_mask.any():
                l_torsion = F.mse_loss(pred_torsion[torsion_mask], target_torsion[torsion_mask])
                l_omega = omega_trans_loss(pred_torsion, torsion_mask)
            else:
                l_torsion = pred_torsion.new_tensor(0.0)
                l_omega = pred_torsion.new_tensor(0.0)

            ss_mask = lig_mask & ss_valid
            if args.ss_label_source == "none":
                ss_mask = torch.zeros_like(lig_mask, dtype=torch.bool)
            if ss_mask.any():
                ss_weights = ss_ce_class_weights(target_ss, ss_mask, args, pred_ss.dtype, pred_ss.device)
                l_ss = F.cross_entropy(pred_ss[ss_mask], target_ss[ss_mask], weight=ss_weights)
                ss_metric_vals = ss_metrics(pred_ss, target_ss, ss_mask)
            else:
                ss_weights = None
                l_ss = pred_torsion.new_tensor(0.0)
                ss_metric_vals = ss_metrics(pred_ss, target_ss, ss_mask)
            target_noncoil_frac = ((target_ss[ss_mask] != 2).float().mean() if ss_mask.any() else pred_torsion.new_tensor(0.0))
            ss_weight_scale = 1.0
            if args.auto_downweight_all_coil_ss and float(target_noncoil_frac.detach().item()) < args.min_ss_noncoil_frac:
                ss_weight_scale = args.all_coil_ss_weight_scale
            l_ss_consistency = ss_consistency_loss(pred_torsion, target_ss, torsion_mask & ss_mask)
            l_target_ss_torsion, l_target_helix_torsion, l_target_beta_torsion = target_ss_torsion_loss(pred_torsion, target_ss, torsion_mask & ss_mask)
            l_ss_smooth = ss_smoothness_loss(pred_ss, lig_mask)

            pocket_center_gt = batch["pocket_center"].to(device, non_blocking=True)
            pocket_radius_gt = batch["pocket_radius"].to(device, non_blocking=True)
            valid_rec = ~rec_pad
            rec_center_ref = (rec_centers * valid_rec.unsqueeze(-1).float()).sum(dim=1) / valid_rec.float().sum(dim=1, keepdim=True).clamp(min=1)
            rec_spread = torch.sqrt(((rec_centers - rec_center_ref[:, None, :]).pow(2).sum(-1) * valid_rec.float()).sum(dim=1) / valid_rec.float().sum(dim=1).clamp(min=1)).unsqueeze(-1).clamp(min=1.0)
            pocket_delta_pred = pocket_pred[:, :3]
            pocket_center_pred = rec_center_ref + pocket_delta_pred
            pocket_radius_pred = F.softplus(pocket_pred[:, 3:4]) + 1e-6
            pocket_center_target_delta = pocket_center_gt - rec_center_ref
            l_pocket_center = F.smooth_l1_loss(pocket_center_pred, pocket_center_gt)
            l_pocket_radius = F.smooth_l1_loss(torch.log(pocket_radius_pred), torch.log(pocket_radius_gt.clamp(min=1e-6)))
            l_pocket_norm = F.smooth_l1_loss(pocket_delta_pred / rec_spread, pocket_center_target_delta / rec_spread)
            l_pocket = l_pocket_center + l_pocket_radius + l_pocket_norm
            pocket_center_error_A = (pocket_center_pred.detach().float() - pocket_center_gt.detach().float()).norm(dim=-1).mean()
            pocket_radius_error_A = (pocket_radius_pred.detach().float() - pocket_radius_gt.detach().float()).abs().mean()

            lig_centers = batch["lig_centers"].to(device, non_blocking=True)
            lig_pad = batch["lig_pad"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=False):
                d_rl = torch.cdist(rec_centers.float(), lig_centers.float())
            valid_pairs = valid_rec.unsqueeze(-1) & (~lig_pad).unsqueeze(1)
            d_rl = d_rl.masked_fill(~valid_pairs, 1e6)
            hotspot_target = (d_rl.min(dim=-1).values <= args.hotspot_cutoff).float()
            hotspot_valid = valid_rec
            if hotspot_valid.any():
                hot_target_valid = hotspot_target[hotspot_valid]
                hot_logits_valid = hotspot_logits[hotspot_valid]
                pos = hot_target_valid.mean().clamp(min=1e-4, max=0.99)
                pos_weight = ((1.0 - pos) / pos).detach()
                l_hotspot = F.binary_cross_entropy_with_logits(
                    hot_logits_valid, hot_target_valid, pos_weight=pos_weight
                )
                hotspot_pos_ratio = hot_target_valid.detach().float().mean()
                hotspot_prob = torch.sigmoid(hot_logits_valid.detach().float())
                hotspot_pred_mean = hotspot_prob.mean()
                hotspot_pred_pos = hotspot_prob >= 0.5
                hotspot_pred_pos_frac = hotspot_pred_pos.float().mean()
                hotspot_true_pos = hot_target_valid.detach().bool()
                tp = (hotspot_pred_pos & hotspot_true_pos).float().sum()
                pred_pos_n = hotspot_pred_pos.float().sum().clamp(min=1.0)
                true_pos_n = hotspot_true_pos.float().sum().clamp(min=1.0)
                hotspot_precision = tp / pred_pos_n
                hotspot_recall = tp / true_pos_n
            else:
                l_hotspot = pred_torsion.new_tensor(0.0)
                hotspot_pos_ratio = pred_torsion.new_tensor(0.0)
                hotspot_pred_mean = pred_torsion.new_tensor(0.0)
                hotspot_pred_pos_frac = pred_torsion.new_tensor(0.0)
                hotspot_precision = pred_torsion.new_tensor(0.0)
                hotspot_recall = pred_torsion.new_tensor(0.0)

            pred_bb = generate_backbone(pred_torsion)
            pred_ca = pred_bb[:, :, 1, :]
            if target_ca.shape[1] != pred_ca.shape[1]:
                target_ca_use = target_ca[:, :pred_ca.shape[1]]
                lig_mask_use = lig_mask[:, :pred_ca.shape[1]]
            else:
                target_ca_use = target_ca
                lig_mask_use = lig_mask
            l_dist = pairwise_distogram_loss(pred_ca, target_ca_use, lig_mask_use)
            l_local_ca = local_ca_distance_loss(pred_ca, target_ca_use, lig_mask_use)
            l_rmsd = ca_rmsd_loss(pred_ca, target_ca_use, lig_mask_use)
            l_kabsch_coord = kabsch_aligned_coord_loss(pred_ca, target_ca_use, lig_mask_use)
            pred_rg = masked_rg(pred_ca, lig_mask_use)
            true_rg = masked_rg(target_ca_use, lig_mask_use)
            l_rg = F.smooth_l1_loss(pred_rg, true_rg) if lig_mask_use.any() else pred_torsion.new_tensor(0.0)
            l_rg_guard = rg_guard_loss(pred_rg, true_rg, args.rg_min_ratio, args.rg_max_ratio) if lig_mask_use.any() else pred_torsion.new_tensor(0.0)
            rg_ratio = (pred_rg.detach().float() / true_rg.detach().float().clamp(min=1e-6)).mean() if lig_mask_use.any() else pred_torsion.new_tensor(0.0)
            l_contact = contact_profile_loss(pred_ca, target_ca_use, rec_centers, rec_pad, lig_mask_use, args.contact_cutoff)
            l_self_clash = ca_self_clash_loss(pred_ca, lig_mask_use, min_dist=args.ca_clash_min_dist)
            l_interface_shape = interface_local_shape_loss(pred_ca, target_ca_use, rec_centers, rec_pad, lig_mask_use, args.interface_shape_cutoff)
            l_ss_geometry_agree = ss_geometry_agreement_loss(pred_bb, pred_ss, lig_mask_use)
            target_ss_use = target_ss[:, :pred_bb.shape[1]]
            ss_geom_mask_use = lig_mask_use & ss_valid[:, :pred_bb.shape[1]]
            l_helix_i4_ca = helix_i4_ca_loss(pred_bb, target_ss_use, ss_geom_mask_use)
            l_helix_hbond = helix_hbond_loss(pred_bb, target_ss_use, ss_geom_mask_use)
            l_beta_extended_geom = beta_extended_geometry_loss(pred_bb, target_ss_use, ss_geom_mask_use)
            l_beta_hbond = beta_hbond_geometry_loss(pred_bb, target_ss_use, ss_geom_mask_use)
            l_beta_pair_ca, l_beta_pair_hbond, l_beta_pair_dir, beta_pair_count, beta_paired_mask = beta_sheet_pairing_losses(
                pred_bb, target_ss_use, ss_geom_mask_use, target_ca_use,
                args.beta_min_segment_len, args.beta_pair_mode, args.beta_pair_orientation,
                args.beta_max_pair_candidates
            )
            l_nonpair_repulsion = nonpair_repulsion_loss(pred_ca, lig_mask_use, beta_paired_mask, args.nonpair_repulsion_min_dist)
            warmup_steps = args.beta_pair_warmup_steps
            if warmup_steps <= 0 and args.beta_pair_warmup_epochs > 0:
                warmup_steps = args.beta_pair_warmup_epochs * max(1, len(loader))
            beta_pair_weight_scale = 1.0
            if warmup_steps > 0:
                beta_pair_weight_scale = min(1.0, float(global_step + 1) / float(max(1, warmup_steps)))
            dssp_metric_vals = dssp_proxy_metrics(pred_bb, pred_ss, target_ss_use, lig_mask_use)
            topology_target = infer_topology_targets(target_ss, lig_mask)
            l_topology = F.cross_entropy(topology_logits, topology_target)

            loss = (
                args.w_torsion * l_torsion + (args.w_ss * ss_weight_scale) * l_ss + args.w_omega * l_omega +
                args.w_pocket * l_pocket + args.w_hotspot * l_hotspot +
                args.w_distogram * l_dist + args.w_local_ca * l_local_ca +
                args.w_ca_rmsd * l_rmsd + args.w_kabsch_coord * l_kabsch_coord +
                args.w_rg * l_rg +
                args.w_contact_profile * l_contact + args.w_topology * l_topology +
                (args.w_ss_consistency * ss_weight_scale) * l_ss_consistency +
                args.w_target_ss_torsion * l_target_helix_torsion +
                args.w_beta_torsion_target * l_target_beta_torsion +
                args.w_ss_smooth * l_ss_smooth +
                args.w_ss_geometry_agree * l_ss_geometry_agree +
                args.w_helix_i4_ca * l_helix_i4_ca + args.w_helix_hbond_target * l_helix_hbond +
                args.w_helix_hbond * l_helix_hbond + args.w_beta_hbond * l_beta_hbond +
                args.w_beta_extended_geom * l_beta_extended_geom +
                beta_pair_weight_scale * (
                    args.w_beta_pair_ca * l_beta_pair_ca + args.w_beta_pair_hbond * l_beta_pair_hbond +
                    args.w_beta_pair_dir * l_beta_pair_dir
                ) +
                args.w_nonpair_repulsion * l_nonpair_repulsion + args.w_rg_guard * l_rg_guard +
                args.w_self_clash * l_self_clash + args.w_interface_shape * l_interface_shape
            )

        if args.amp:
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

        total_loss += float(loss.item())
        step_metrics = {
            "loss": float(loss.item()),
            "tor": float(l_torsion.item()),
            "ss": float(l_ss.item()),
            "ss_consistency": float(l_ss_consistency.item()),
            "ss_smooth": float(l_ss_smooth.item()),
            "ss_geometry_agree": float(l_ss_geometry_agree.item()),
            "target_ss_torsion": float(l_target_ss_torsion.item()),
            "target_helix_torsion": float(l_target_helix_torsion.item()),
            "target_beta_torsion": float(l_target_beta_torsion.item()),
            "helix_i4_ca": float(l_helix_i4_ca.item()),
            "helix_hbond": float(l_helix_hbond.item()),
            "beta_hbond": float(l_beta_hbond.item()),
            "beta_extended_geom": float(l_beta_extended_geom.item()),
            "beta_pair_ca": float(l_beta_pair_ca.item()),
            "beta_pair_hbond": float(l_beta_pair_hbond.item()),
            "beta_pair_dir": float(l_beta_pair_dir.item()),
            "beta_pair_count": float(beta_pair_count.item()),
            "beta_pair_weight_scale": float(beta_pair_weight_scale),
            "nonpair_repulsion": float(l_nonpair_repulsion.item()),
            "rg_guard": float(l_rg_guard.item()),
            "ss_ce_weight_helix": float(ss_weights[0].detach().item()) if ss_weights is not None else 1.0,
            "ss_ce_weight_beta": float(ss_weights[1].detach().item()) if ss_weights is not None else 1.0,
            "ss_ce_weight_coil": float(ss_weights[2].detach().item()) if ss_weights is not None else 1.0,
            "ss_weight_scale": float(ss_weight_scale),
            "ss_target_noncoil_frac": float(target_noncoil_frac.detach().item()),
            "ss_valid_ratio": float((ss_mask.float().sum() / lig_mask.float().sum().clamp(min=1.0)).detach().item()),
            "ss_source_dssp_frac": float(sum(1 for x in ss_sources if x == "dssp") / max(1, len(ss_sources))) if isinstance(ss_sources, list) else 0.0,
            **ss_metric_vals,
            **dssp_metric_vals,
            "omega": float(l_omega.item()),
            "pocket": float(l_pocket.item()),
            "pocket_center_loss": float(l_pocket_center.item()),
            "pocket_radius_loss": float(l_pocket_radius.item()),
            "pocket_center_error_A": float(pocket_center_error_A.item()),
            "pocket_radius_error_A": float(pocket_radius_error_A.item()),
            "hotspot_loss": float(l_hotspot.item()),
            "hotspot_pos_ratio": float(hotspot_pos_ratio.item()),
            "hotspot_pred_mean": float(hotspot_pred_mean.item()),
            "hotspot_pred_pos_frac": float(hotspot_pred_pos_frac.item()),
            "hotspot_precision": float(hotspot_precision.item()),
            "hotspot_recall": float(hotspot_recall.item()),
            "rmsd": float(l_rmsd.item()),
            "dist": float(l_dist.item()),
            "local_ca": float(l_local_ca.item()),
            "kabsch_coord": float(l_kabsch_coord.item()),
            "rg": float(l_rg.item()),
            "rg_ratio": float(rg_ratio.item()),
            "contact": float(l_contact.item()),
            "self_clash": float(l_self_clash.item()),
            "interface_shape": float(l_interface_shape.item()),
            "topology": float(l_topology.item()),
        }
        for k, v in step_metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v
        n_steps += 1
        global_step += 1
        denom_steps = max(1, n_steps)
        pbar.set_postfix({
            "loss": f"{metric_sums['loss'] / denom_steps:.4f}",
            "tor": f"{l_torsion.item():.4f}",
            "ss": f"{l_ss.item():.3f}",
            "omg": f"{l_omega.item():.3f}",
            "pcE": f"{pocket_center_error_A.item():.1f}A",
            "prE": f"{pocket_radius_error_A.item():.1f}A",
            "hot": f"{l_hotspot.item():.3f}",
            "hpos": f"{hotspot_pos_ratio.item():.2f}",
            "hrec": f"{hotspot_recall.item():.2f}",
            "rmsd": f"{l_rmsd.item():.2f}",
            "dist": f"{l_dist.item():.2f}",
            "top": f"{l_topology.item():.2f}",
        })
        if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
            print(
                f"[Stage4][Step {global_step}] "
                f"epoch={epoch} iter={it} "
                f"loss={loss.item():.4f} tor={l_torsion.item():.4f} "
                f"ss={l_ss.item():.4f} ss_acc={ss_metric_vals.get('ss_acc', 0.0):.4f} "
                f"ss_source={','.join(ss_sources) if isinstance(ss_sources, list) else args.ss_label_source} "
                f"ss_valid={step_metrics.get('ss_valid_ratio', 0.0):.4f} "
                f"ss_scale={ss_weight_scale:.3f} ss_noncoil={target_noncoil_frac.item():.4f} "
                f"ss_consistency={l_ss_consistency.item():.4f} ss_smooth={l_ss_smooth.item():.4f} "
                f"target_ss_tor={l_target_ss_torsion.item():.4f} htor={l_target_helix_torsion.item():.4f} btor={l_target_beta_torsion.item():.4f} "
                f"ss_geom={l_ss_geometry_agree.item():.4f} helix_i4={l_helix_i4_ca.item():.4f} helix_hbond={l_helix_hbond.item():.4f} "
                f"beta_ext={l_beta_extended_geom.item():.4f} beta_hbond={l_beta_hbond.item():.4f} "
                f"beta_pair_ca={l_beta_pair_ca.item():.4f} beta_pair_hbond={l_beta_pair_hbond.item():.4f} "
                f"beta_pair_dir={l_beta_pair_dir.item():.4f} beta_pair_count={beta_pair_count.item():.0f} "
                f"beta_pair_w={beta_pair_weight_scale:.3f} nonpair_rep={l_nonpair_repulsion.item():.4f} rg_guard={l_rg_guard.item():.4f} "
                f"dssp_proxy_acc={dssp_metric_vals.get('dssp_proxy_acc', 0.0):.4f} "
                f"omega={l_omega.item():.4f} "
                f"pocket_center_error_A={pocket_center_error_A.item():.3f} "
                f"pocket_radius_error_A={pocket_radius_error_A.item():.3f} "
                f"aligned_rmsd={l_rmsd.item():.4f} dist_error={l_dist.item():.4f} "
                f"local_ca={l_local_ca.item():.4f} "
                f"kabsch_coord={l_kabsch_coord.item():.4f} "
                f"rg_ratio={rg_ratio.item():.4f} "
                f"hotspot_precision={hotspot_precision.item():.4f} "
                f"hotspot_recall={hotspot_recall.item():.4f} "
                f"contact_profile={l_contact.item():.4f} "
                f"self_clash={l_self_clash.item():.4f} "
                f"interface_shape={l_interface_shape.item():.4f} "
                f"topology={l_topology.item():.4f}"
            )

    avg_metrics = {k: v / max(1, n_steps) for k, v in metric_sums.items()}
    if avg_metrics:
        print(
            f"[Stage4][Epoch {epoch} Metrics] "
            f"loss={avg_metrics.get('loss', 0.0):.4f} "
            f"tor={avg_metrics.get('tor', 0.0):.4f} ss={avg_metrics.get('ss', 0.0):.4f} "
            f"ss_acc={avg_metrics.get('ss_acc', 0.0):.4f} "
            f"ss_source={args.ss_label_source} ss_source_dssp_frac={avg_metrics.get('ss_source_dssp_frac', 0.0):.4f} "
            f"ss_valid_ratio={avg_metrics.get('ss_valid_ratio', 0.0):.4f} "
            f"ss_consistency={avg_metrics.get('ss_consistency', 0.0):.4f} "
            f"ss_smooth={avg_metrics.get('ss_smooth', 0.0):.4f} "
            f"ss_geometry_agree={avg_metrics.get('ss_geometry_agree', 0.0):.4f} "
            f"target_ss_torsion={avg_metrics.get('target_ss_torsion', 0.0):.4f} "
            f"target_helix_torsion={avg_metrics.get('target_helix_torsion', 0.0):.4f} "
            f"target_beta_torsion={avg_metrics.get('target_beta_torsion', 0.0):.4f} "
            f"helix_i4_ca={avg_metrics.get('helix_i4_ca', 0.0):.4f} "
            f"helix_hbond={avg_metrics.get('helix_hbond', 0.0):.4f} "
            f"beta_hbond={avg_metrics.get('beta_hbond', 0.0):.4f} "
            f"beta_extended_geom={avg_metrics.get('beta_extended_geom', 0.0):.4f} "
            f"beta_pair_ca={avg_metrics.get('beta_pair_ca', 0.0):.4f} "
            f"beta_pair_hbond={avg_metrics.get('beta_pair_hbond', 0.0):.4f} "
            f"beta_pair_dir={avg_metrics.get('beta_pair_dir', 0.0):.4f} "
            f"beta_pair_count={avg_metrics.get('beta_pair_count', 0.0):.1f} "
            f"beta_pair_w={avg_metrics.get('beta_pair_weight_scale', 1.0):.3f} "
            f"nonpair_repulsion={avg_metrics.get('nonpair_repulsion', 0.0):.4f} "
            f"rg_guard={avg_metrics.get('rg_guard', 0.0):.4f} "
            f"dssp_proxy_acc={avg_metrics.get('dssp_proxy_acc', 0.0):.4f} "
            f"dssp_proxy_target_acc={avg_metrics.get('dssp_proxy_target_acc', 0.0):.4f} "
            f"dssp_proxy_frac=({avg_metrics.get('dssp_proxy_helix_frac', 0.0):.3f},"
            f"{avg_metrics.get('dssp_proxy_beta_frac', 0.0):.3f},"
            f"{avg_metrics.get('dssp_proxy_coil_frac', 0.0):.3f}) "
            f"ss_helix_precision={avg_metrics.get('ss_helix_precision', 0.0):.4f} "
            f"ss_helix_recall={avg_metrics.get('ss_helix_recall', 0.0):.4f} "
            f"ss_beta_precision={avg_metrics.get('ss_beta_precision', 0.0):.4f} "
            f"ss_beta_recall={avg_metrics.get('ss_beta_recall', 0.0):.4f} "
            f"ss_coil_precision={avg_metrics.get('ss_coil_precision', 0.0):.4f} "
            f"ss_coil_recall={avg_metrics.get('ss_coil_recall', 0.0):.4f} "
            f"ss_macro_f1={avg_metrics.get('ss_macro_f1', 0.0):.4f} "
            f"ss_balanced_acc={avg_metrics.get('ss_balanced_acc', 0.0):.4f} "
            f"ss_ce_weights=({avg_metrics.get('ss_ce_weight_helix', 1.0):.3f},"
            f"{avg_metrics.get('ss_ce_weight_beta', 1.0):.3f},"
            f"{avg_metrics.get('ss_ce_weight_coil', 1.0):.3f}) "
            f"ss_pred_frac=({avg_metrics.get('ss_pred_helix_frac', 0.0):.3f},"
            f"{avg_metrics.get('ss_pred_beta_frac', 0.0):.3f},"
            f"{avg_metrics.get('ss_pred_coil_frac', 0.0):.3f}) "
            f"ss_target_frac=({avg_metrics.get('ss_target_helix_frac', 0.0):.3f},"
            f"{avg_metrics.get('ss_target_beta_frac', 0.0):.3f},"
            f"{avg_metrics.get('ss_target_coil_frac', 0.0):.3f}) "
            f"omega={avg_metrics.get('omega', 0.0):.4f} "
            f"pocket={avg_metrics.get('pocket', 0.0):.4f} "
            f"pocket_center_loss={avg_metrics.get('pocket_center_loss', 0.0):.4f} "
            f"pocket_radius_loss={avg_metrics.get('pocket_radius_loss', 0.0):.4f} "
            f"pocket_center_error_A={avg_metrics.get('pocket_center_error_A', 0.0):.3f} "
            f"pocket_radius_error_A={avg_metrics.get('pocket_radius_error_A', 0.0):.3f} "
            f"hotspot_loss={avg_metrics.get('hotspot_loss', 0.0):.4f} "
            f"hotspot_pos_ratio={avg_metrics.get('hotspot_pos_ratio', 0.0):.4f} "
            f"hotspot_pred_mean={avg_metrics.get('hotspot_pred_mean', 0.0):.4f} "
            f"hotspot_pred_pos_frac={avg_metrics.get('hotspot_pred_pos_frac', 0.0):.4f} "
            f"hotspot_precision={avg_metrics.get('hotspot_precision', 0.0):.4f} "
            f"hotspot_recall={avg_metrics.get('hotspot_recall', 0.0):.4f} "
            f"rmsd={avg_metrics.get('rmsd', 0.0):.4f} "
            f"dist={avg_metrics.get('dist', 0.0):.4f} "
            f"local_ca={avg_metrics.get('local_ca', 0.0):.4f} "
            f"kabsch_coord={avg_metrics.get('kabsch_coord', 0.0):.4f} "
            f"rg={avg_metrics.get('rg', 0.0):.4f} "
            f"rg_ratio={avg_metrics.get('rg_ratio', 0.0):.4f} "
            f"contact={avg_metrics.get('contact', 0.0):.4f} "
            f"self_clash={avg_metrics.get('self_clash', 0.0):.4f} "
            f"interface_shape={avg_metrics.get('interface_shape', 0.0):.4f} "
            f"topology={avg_metrics.get('topology', 0.0):.4f}"
        )

    return total_loss / max(1, n_steps), global_step, avg_metrics


def load_stage3_encoder(model: TorsionGenerator, ckpt_path: str, device):
    if ckpt_path and os.path.isfile(ckpt_path):
        model.load_from_stage3(ckpt_path, device=device)


def save_train_mode_predicted_pdb(model, loader, device, output_pdb: str):
    model.eval()
    batch = next(iter(loader))
    rec_feats = batch["rec_feats"].to(device, non_blocking=True)
    rec_centers = batch["rec_centers"].to(device, non_blocking=True)
    rec_pad = batch["rec_pad"].to(device, non_blocking=True)
    target_torsion = batch["lig_torsion_target"].to(device, non_blocking=True)
    L = min(target_torsion.shape[1], getattr(model, "max_residues", target_torsion.shape[1]))
    with torch.no_grad():
        pred_torsion, _pred_ss, _pocket_pred, _hotspot_logits, _topology_logits = model(
            rec_feats, rec_centers, rec_pad, num_residues=L
        )
        pred_torsion = normalize_torsion_sincos(pred_torsion)
        backbone = generate_backbone(pred_torsion)
    bb_np = backbone[0].detach().cpu().numpy()
    report = validate_backbone(bb_np)
    os.makedirs(os.path.dirname(output_pdb) or ".", exist_ok=True)
    write_pdb(bb_np, output_pdb)
    print(f"[Stage4][TrainModePDB] wrote {output_pdb}")
    print(
        f"[Stage4][TrainModePDB] ca_ca_mean={report['ca_ca_mean']:.3f} "
        f"ca_ca_std={report['ca_ca_std']:.3f} clashes={report['n_clashes']} residues={report['n_residues']}"
    )


def main():
    ap = argparse.ArgumentParser(description="Train Stage4 torsion generator")
    ap.add_argument("--skempi_csv", type=str,
                    default="/data2/jiangjiaqi/srzhang/InversionDock/Data/Skempi_dataset/skempi_v2.csv")
    ap.add_argument("--npz_root", type=str,
                    default="/data2/jiangjiaqi/srzhang/InversionDock/Data/Processed_skempi_backbone_aware")
    ap.add_argument("--stage3_ckpt", type=str, default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_residues", type=int, default=128)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers_surf", type=int, default=6)
    ap.add_argument("--nlayers_gen", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--surface_sampling", type=str, default="interface", choices=["random", "interface"],
                    help="Use interface-centered receptor/ligand patches for Stage 4 training")
    ap.add_argument("--pocket_target_type", type=str, default="surface_midpoint_robust",
                    choices=["surface_midpoint_robust", "native_ligand_ca", "native_interface_ca"],
                    help="Pocket target definition for Stage4 pocket head ablations")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,
                    help="If >0, stop each epoch after this many optimizer steps for quick tests")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--w_torsion", type=float, default=10.0)
    ap.add_argument("--w_ss", type=float, default=3.0)
    ap.add_argument("--w_omega", type=float, default=1.0)
    ap.add_argument("--w_pocket", type=float, default=5.0)
    ap.add_argument("--w_hotspot", type=float, default=6.0)
    ap.add_argument("--w_distogram", type=float, default=3.0)
    ap.add_argument("--w_local_ca", type=float, default=1.0,
                    help="Local CA distance loss over i,i+2/i+3/i+4/i+5 to improve compact protein-like shape")
    ap.add_argument("--w_ca_rmsd", type=float, default=1.0)
    ap.add_argument("--w_kabsch_coord", type=float, default=0.0,
                    help="SmoothL1 coordinate loss after Kabsch alignment; shape-overfit only")
    ap.add_argument("--w_rg", type=float, default=0.75)
    ap.add_argument("--w_contact_profile", type=float, default=2.0)
    ap.add_argument("--w_topology", type=float, default=1.0)
    ap.add_argument("--w_ss_consistency", type=float, default=0.0,
                    help="Weak torsion/SS consistency loss for helix and beta targets; coil is unconstrained")
    ap.add_argument("--w_target_ss_torsion", type=float, default=0.0,
                    help="Target-DSSP-driven helix/beta torsion prior on predicted torsions")
    ap.add_argument("--w_helix_i4_ca", type=float, default=0.0,
                    help="Target helix CA(i)-CA(i+4) distance geometry loss")
    ap.add_argument("--w_helix_hbond_target", type=float, default=0.0,
                    help="Target helix i->i+4 approximate O-N H-bond geometry loss")
    ap.add_argument("--w_beta_torsion_target", type=float, default=0.0,
                    help="Extra weight on target beta phi/psi torsion prior")
    ap.add_argument("--w_beta_extended_geom", type=float, default=0.0,
                    help="Target beta extended-strand local CA distance geometry loss")
    ap.add_argument("--w_beta_pair_ca", type=float, default=0.0,
                    help="Target beta strand-pair CA-CA distance loss")
    ap.add_argument("--w_beta_pair_hbond", type=float, default=0.0,
                    help="Target beta strand-pair approximate O-N H-bond loss")
    ap.add_argument("--w_beta_pair_dir", type=float, default=0.0,
                    help="Target beta strand-pair parallel/antiparallel direction consistency loss")
    ap.add_argument("--w_beta_scaffold_ca", type=float, default=0.0,
                    help="Compatibility no-op: beta scaffold CA loss is disabled in this trainer")
    ap.add_argument("--w_beta_scaffold_pair", type=float, default=0.0,
                    help="Compatibility no-op: beta scaffold pair loss is disabled in this trainer")
    ap.add_argument("--w_beta_scaffold_shape", type=float, default=0.0,
                    help="Compatibility no-op: beta scaffold shape loss is disabled in this trainer")
    ap.add_argument("--w_beta_scaffold_hbond", type=float, default=0.0,
                    help="Compatibility no-op: beta scaffold H-bond loss is disabled in this trainer")
    ap.add_argument("--beta_min_segment_len", type=int, default=3,
                    help="Minimum target beta segment length for strand-pair supervision")
    ap.add_argument("--beta_pair_mode", type=str, default="native_ca", choices=["native_ca", "sequence_heuristic"],
                    help="How to choose target beta strand-pair candidates")
    ap.add_argument("--beta_pair_orientation", type=str, default="both", choices=["both", "parallel", "antiparallel"],
                    help="Allowed beta strand-pair orientations")
    ap.add_argument("--beta_max_pair_candidates", type=int, default=4,
                    help="Maximum target beta strand-pair candidates per sample")
    ap.add_argument("--beta_pair_warmup_epochs", type=int, default=0,
                    help="Linearly warm beta pair losses over this many epochs")
    ap.add_argument("--beta_pair_warmup_steps", type=int, default=0,
                    help="Linearly warm beta pair losses over this many global steps; overrides epoch-derived steps if >0")
    ap.add_argument("--w_nonpair_repulsion", type=float, default=0.0,
                    help="Repel nonlocal, non-paired CA pairs to prevent beta-pair collapse")
    ap.add_argument("--nonpair_repulsion_min_dist", type=float, default=4.0)
    ap.add_argument("--w_rg_guard", type=float, default=0.0,
                    help="Penalty for predicted Rg ratio outside [rg_min_ratio, rg_max_ratio]")
    ap.add_argument("--rg_min_ratio", type=float, default=0.85)
    ap.add_argument("--rg_max_ratio", type=float, default=1.3)
    ap.add_argument("--w_ss_geometry_agree", type=float, default=0.0,
                    help="Low-weight diagnostic regularizer: SS logits match generated-geometry DSSP proxy")
    ap.add_argument("--w_helix_hbond", type=float, default=0.0,
                    help="Explicit alpha-helix i->i+4 backbone H-bond geometry loss on helix targets")
    ap.add_argument("--w_beta_hbond", type=float, default=0.0,
                    help="Explicit long-range backbone H-bond geometry loss on beta targets")
    ap.add_argument("--w_ss_smooth", type=float, default=0.0,
                    help="Small adjacent-residue smoothness penalty on SS probabilities")
    ap.add_argument("--w_self_clash", type=float, default=0.0,
                    help="CA-level nonlocal self-clash penalty for generated backbone shape")
    ap.add_argument("--w_interface_shape", type=float, default=0.0,
                    help="Interface-local distogram/aligned-coordinate shape loss")
    ap.add_argument("--ca_clash_min_dist", type=float, default=2.8)
    ap.add_argument("--interface_shape_cutoff", type=float, default=8.0)
    ap.add_argument("--ss_label_source", type=str, default="auto", choices=["auto", "dssp", "clean", "original", "none"],
                    help="Secondary-structure target source; auto prefers DSSP labels when valid, otherwise falls back to clean/original")
    ap.add_argument("--auto_downweight_all_coil_ss", action="store_true",
                    help="Scale SS CE/consistency down when target non-coil fraction is too small")
    ap.add_argument("--min_ss_noncoil_frac", type=float, default=0.02)
    ap.add_argument("--all_coil_ss_weight_scale", type=float, default=0.1)
    ap.add_argument("--smooth_ss_labels", action="store_true",
                    help="Convert helix/beta segments shorter than --ss_min_segment_len to coil in dataset targets")
    ap.add_argument("--ss_min_segment_len", type=int, default=3,
                    help="Minimum helix/beta segment length retained when --smooth_ss_labels is enabled")
    ap.add_argument("--ss_class_weight", type=str, default="none", choices=["none", "auto", "manual"],
                    help="Use class-balanced SS cross entropy weights to counter coil dominance")
    ap.add_argument("--ss_weight_helix", type=float, default=2.0)
    ap.add_argument("--ss_weight_beta", type=float, default=1.5)
    ap.add_argument("--ss_weight_coil", type=float, default=0.7)
    ap.add_argument("--contact_cutoff", type=float, default=8.0)
    ap.add_argument("--hotspot_cutoff", type=float, default=8.0)
    ap.add_argument("--init_ckpt", type=str, default="", help="Optional Stage4 checkpoint to fine-tune from")
    ap.add_argument("--save_dir", type=str, default="./ckpts_stage4_generator")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overfit_n", type=int, default=0,
                    help="If >0, restrict training to the first N samples for small-batch overfit debugging")
    ap.add_argument("--overfit_repeat", type=int, default=0,
                    help="If >0, repeat the selected overfit subset this many times per epoch")
    ap.add_argument("--overfit_samples_json", type=str, default="",
                    help="Optional fixed sample JSON produced for Stage4 overfit debugging")
    ap.add_argument("--train_samples_json", type=str, default="",
                    help="Optional fixed Stage4 training sample JSON; training is restricted exactly to these samples")
    ap.add_argument("--log_every_steps", type=int, default=500,
                    help="Print detailed global-step diagnostics every N optimizer steps")
    ap.add_argument("--save_train_pred_pdb", type=str, default="",
                    help="If set with --init_ckpt, save train-mode predicted backbone PDB for the first batch and exit")
    ap.add_argument("--train_pocket_head_only", action="store_true",
                    help="Freeze all parameters except pocket_head for placement-only fine-tuning")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    ds = SkempiDockingDataset(
        args.skempi_csv, args.npz_root, K=args.K,
        seq_len=args.seq_len, max_residues=args.max_residues,
        cache_npz=True, surface_sampling=args.surface_sampling,
        pocket_target_type=args.pocket_target_type,
        smooth_ss_labels=args.smooth_ss_labels,
        ss_min_segment_len=args.ss_min_segment_len,
        ss_label_source=args.ss_label_source,
    )
    if args.train_samples_json:
        with open(args.train_samples_json, "r") as f:
            train_payload = json.load(f)
        fixed_samples = train_payload.get("samples", train_payload if isinstance(train_payload, list) else [])
        sample_map = {}
        for row_idx, rec, lig, pdb_str in ds.samples:
            rec_abs = os.path.abspath(rec)
            lig_abs = os.path.abspath(lig)
            sample_map[(rec_abs, lig_abs)] = (row_idx, rec, lig, pdb_str)
            sample_map[(os.path.basename(rec), os.path.basename(lig))] = (row_idx, rec, lig, pdb_str)
        selected = []
        missing = []
        for item in fixed_samples:
            key_abs = (os.path.abspath(item["rec_npz"]), os.path.abspath(item["lig_npz"]))
            key_base = (os.path.basename(item["rec_npz"]), os.path.basename(item["lig_npz"]))
            if key_abs in sample_map:
                selected.append(sample_map[key_abs])
            elif key_base in sample_map:
                selected.append(sample_map[key_base])
            else:
                missing.append(item.get("sample_id", str(key_abs)))
        if missing:
            raise RuntimeError(f"Fixed training samples not found in dataset: {missing[:20]} (n_missing={len(missing)})")
        if not selected:
            raise RuntimeError(f"No samples selected from --train_samples_json={args.train_samples_json}")
        ds.samples = selected
        print(f"[Stage4][TrainSamples] using {len(selected)} fixed samples from {args.train_samples_json}")
    if args.overfit_n > 0:
        overfit_n = min(args.overfit_n, len(ds))
        if args.overfit_samples_json:
            with open(args.overfit_samples_json, "r") as f:
                overfit_payload = json.load(f)
            fixed_samples = overfit_payload.get(f"overfit_n{overfit_n}") or overfit_payload.get("samples", [])[:overfit_n]
            sample_map = {}
            for row_idx, rec, lig, pdb_str in ds.samples:
                rec_abs = os.path.abspath(rec)
                lig_abs = os.path.abspath(lig)
                sample_map[(rec_abs, lig_abs)] = (row_idx, rec, lig, pdb_str)
                sample_map[(os.path.basename(rec), os.path.basename(lig))] = (row_idx, rec, lig, pdb_str)
            selected = []
            missing = []
            for item in fixed_samples[:overfit_n]:
                key_abs = (os.path.abspath(item["rec_npz"]), os.path.abspath(item["lig_npz"]))
                key_base = (os.path.basename(item["rec_npz"]), os.path.basename(item["lig_npz"]))
                if key_abs in sample_map:
                    selected.append(sample_map[key_abs])
                elif key_base in sample_map:
                    selected.append(sample_map[key_base])
                else:
                    missing.append(item.get("sample_id", str(key_abs)))
            if missing:
                raise RuntimeError(f"Fixed overfit samples not found in dataset: {missing}")
            ds.samples = selected
            print(f"[Stage4][Overfit] using {len(selected)} fixed samples from {args.overfit_samples_json}")
        else:
            ds.samples = ds.samples[:overfit_n]
            print(f"[Stage4][Overfit] restricting dataset to first {overfit_n} samples")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    drop_last=False, collate_fn=stage3_collate_fn)
    print(f"[Stage4][DataLoader] dataset_size={len(ds)} batch_size={args.batch_size} steps_per_epoch={len(dl)} drop_last=False")
    if args.overfit_n > 0 and args.overfit_repeat > 1:
        cached = []
        base_iter = iter(dl)
        for _ in range(args.overfit_repeat):
            try:
                cached.append(next(base_iter))
            except StopIteration:
                base_iter = iter(dl)
                cached.append(next(base_iter))
        dl = cached
        print(f"[Stage4][Overfit] overfit_repeat={args.overfit_repeat}; steps_per_epoch={len(dl)}; total_planned_steps={len(dl) * args.epochs}")

    model = TorsionGenerator(
        d_model=args.d_model, nhead=args.nhead,
        nlayers_surf=args.nlayers_surf, nlayers_gen=args.nlayers_gen,
        K=args.K, max_residues=args.max_residues, dropout=args.dropout,
    ).to(device)
    load_stage3_encoder(model, args.stage3_ckpt, device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        load_compatible_state_dict(model, ckpt['model'], label=args.init_ckpt)

    if args.train_pocket_head_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.pocket_head.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Stage4][PocketOnly] trainable_params={trainable} / {total_params}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if args.save_train_pred_pdb:
        if not args.init_ckpt:
            raise RuntimeError("--save_train_pred_pdb requires --init_ckpt")
        save_train_mode_predicted_pdb(model, dl, device, args.save_train_pred_pdb)
        return

    global_step = 0
    last_epoch_metrics = {}
    for epoch in range(args.epochs):
        avg, global_step, epoch_metrics = train_one_epoch(model, dl, optimizer, scaler, device, epoch, args, global_step=global_step)
        last_epoch_metrics = epoch_metrics
        print(f"[Stage4][Epoch {epoch}] avg_loss={avg:.4f} global_step={global_step}")
        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"e{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(), "args": vars(args)}, path)
            print(f"[Stage4] Saved {path}")

    final = os.path.join(args.save_dir, "final.pt")
    final_payload = {"epoch": args.epochs - 1, "model": model.state_dict(), "optim": optimizer.state_dict(), "args": vars(args), "train_metrics": last_epoch_metrics}
    torch.save(final_payload, final)
    with open(os.path.join(args.save_dir, "final_metrics.json"), "w") as f:
        json.dump(last_epoch_metrics, f, indent=2, sort_keys=True)
    print(f"[Stage4] Done. Final checkpoint: {final}")


if __name__ == "__main__":
    main()
