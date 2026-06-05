# -*- coding: utf-8 -*-
"""
Backbone geometry loss functions for protein structure generation.
All losses operate on backbone tensors of shape (B, L, 3, 3) where dim -2 is [N, CA, C].
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import (
    BOND_N_CA, BOND_CA_C, BOND_C_N,
    ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_C_N_CA,
    OMEGA_TRANS, HELIX_PHI, HELIX_PSI, BETA_PHI, BETA_PSI,
    compute_dihedral_torch, legacy_omega_to_trans180_torch,
)


def _ensure_4d(backbone: torch.Tensor) -> torch.Tensor:
    if backbone.dim() == 3:
        return backbone.unsqueeze(0)
    return backbone


def bond_length_loss(backbone: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from ideal N-CA, CA-C, C-N bond lengths."""
    backbone = _ensure_4d(backbone)
    N = backbone[:, :, 0, :]
    CA = backbone[:, :, 1, :]
    C = backbone[:, :, 2, :]

    d_n_ca = (CA - N).norm(dim=-1)
    d_ca_c = (C - CA).norm(dim=-1)
    d_c_n = (N[:, 1:] - C[:, :-1]).norm(dim=-1)

    loss_n_ca = F.mse_loss(d_n_ca, torch.full_like(d_n_ca, BOND_N_CA))
    loss_ca_c = F.mse_loss(d_ca_c, torch.full_like(d_ca_c, BOND_CA_C))
    loss_c_n = F.mse_loss(d_c_n, torch.full_like(d_c_n, BOND_C_N))

    return loss_n_ca + loss_ca_c + loss_c_n


def bond_angle_loss(backbone: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from ideal N-CA-C, CA-C-N, C-N-CA bond angles."""
    backbone = _ensure_4d(backbone)
    B, L, _, _ = backbone.shape
    N = backbone[:, :, 0, :]
    CA = backbone[:, :, 1, :]
    C = backbone[:, :, 2, :]

    def angle_between(v1, v2):
        cos_a = (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1)).clamp(min=1e-8)
        return torch.acos(cos_a.clamp(-1 + 1e-7, 1 - 1e-7)) * (180.0 / math.pi)

    v1 = N - CA
    v2 = C - CA
    a_n_ca_c = angle_between(v1, v2)
    loss_1 = F.mse_loss(a_n_ca_c, torch.full_like(a_n_ca_c, ANGLE_N_CA_C))

    if L > 1:
        v1 = CA[:, :-1] - C[:, :-1]
        v2 = N[:, 1:] - C[:, :-1]
        a_ca_c_n = angle_between(v1, v2)
        loss_2 = F.mse_loss(a_ca_c_n, torch.full_like(a_ca_c_n, ANGLE_CA_C_N))

        v1 = C[:, :-1] - N[:, 1:]
        v2 = CA[:, 1:] - N[:, 1:]
        a_c_n_ca = angle_between(v1, v2)
        loss_3 = F.mse_loss(a_c_n_ca, torch.full_like(a_c_n_ca, ANGLE_C_N_CA))
    else:
        loss_2 = torch.tensor(0.0, device=backbone.device)
        loss_3 = torch.tensor(0.0, device=backbone.device)

    return loss_1 + loss_2 + loss_3


def omega_loss(backbone: torch.Tensor) -> torch.Tensor:
    """Penalize deviation of omega from 180 degrees (trans peptide bond)."""
    backbone = _ensure_4d(backbone)
    B, L, _, _ = backbone.shape
    if L < 2:
        return torch.tensor(0.0, device=backbone.device)

    CA = backbone[:, :, 1, :]
    C = backbone[:, :, 2, :]
    N = backbone[:, :, 0, :]

    omega = compute_dihedral_torch(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])
    omega = legacy_omega_to_trans180_torch(omega)

    diff = omega - OMEGA_TRANS
    diff = (diff + 180.0) % 360.0 - 180.0
    return (diff ** 2).mean()


def ramachandran_loss(backbone: torch.Tensor, ss_target: torch.Tensor = None) -> torch.Tensor:
    """
    Soft Ramachandran constraint. Penalizes phi/psi outside allowed regions.
    ss_target: (B, L) int tensor, 0=H, 1=E, 2=C. If None, uses general allowed region.
    """
    backbone = _ensure_4d(backbone)
    B, L, _, _ = backbone.shape
    if L < 2:
        return torch.tensor(0.0, device=backbone.device)

    N = backbone[:, :, 0, :]
    CA = backbone[:, :, 1, :]
    C = backbone[:, :, 2, :]

    phi = compute_dihedral_torch(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])
    psi = compute_dihedral_torch(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])

    def angular_diff(a, b):
        d = a - b
        return (d + 180.0) % 360.0 - 180.0

    d_helix = angular_diff(phi, HELIX_PHI) ** 2 + angular_diff(psi, HELIX_PSI) ** 2
    d_beta = angular_diff(phi, BETA_PHI) ** 2 + angular_diff(psi, BETA_PSI) ** 2

    if ss_target is not None:
        ss = ss_target[:, 1:]
        helix_mask = (ss == 0).float()
        beta_mask = (ss == 1).float()
        coil_mask = (ss == 2).float()

        loss_helix = (d_helix * helix_mask).sum() / helix_mask.sum().clamp(min=1)
        loss_beta = (d_beta * beta_mask).sum() / beta_mask.sum().clamp(min=1)
        min_dist = torch.min(d_helix, d_beta)
        threshold = 3600.0
        loss_coil = (F.relu(min_dist - threshold) * coil_mask).sum() / coil_mask.sum().clamp(min=1)
        return loss_helix + loss_beta + loss_coil
    else:
        min_dist = torch.min(d_helix, d_beta)
        threshold = 3600.0
        violation = F.relu(min_dist - threshold)
        return violation.mean()


def clash_loss(backbone: torch.Tensor, min_dist: float = 2.8) -> torch.Tensor:
    """Penalize steric clashes between non-adjacent CA atoms."""
    backbone = _ensure_4d(backbone)
    B, L, _, _ = backbone.shape
    if L < 4:
        return torch.tensor(0.0, device=backbone.device)

    CA = backbone[:, :, 1, :]

    total = torch.tensor(0.0, device=backbone.device)
    for b in range(B):
        dmat = torch.cdist(CA[b:b+1], CA[b:b+1]).squeeze(0)
        idx = torch.arange(L, device=backbone.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = (sep >= 3) & torch.triu(torch.ones(L, L, dtype=torch.bool, device=backbone.device), diagonal=1)
        if mask.any():
            violations = F.relu(min_dist - dmat[mask])
            total = total + (violations ** 2).mean()

    return total / B


def ca_ca_spacing_loss(backbone: torch.Tensor, target: float = 3.80) -> torch.Tensor:
    backbone = _ensure_4d(backbone)
    CA = backbone[:, :, 1, :]
    if CA.shape[1] < 2:
        return torch.tensor(0.0, device=backbone.device)
    d = (CA[:, 1:] - CA[:, :-1]).norm(dim=-1)
    return F.mse_loss(d, torch.full_like(d, target))


def combined_backbone_loss(backbone: torch.Tensor,
                           ss_target: torch.Tensor = None,
                           w_bond: float = 100.0,
                           w_angle: float = 50.0,
                           w_omega: float = 20.0,
                           w_rama: float = 10.0,
                           w_clash: float = 5.0,
                           w_ca_ca: float = 20.0) -> dict:
    """Compute all backbone losses and return weighted total + individual terms."""
    l_bond = bond_length_loss(backbone)
    l_angle = bond_angle_loss(backbone)
    l_omega = omega_loss(backbone)
    l_rama = ramachandran_loss(backbone, ss_target)
    l_clash = clash_loss(backbone)
    l_ca_ca = ca_ca_spacing_loss(backbone)

    total = (w_bond * l_bond + w_angle * l_angle + w_omega * l_omega +
             w_rama * l_rama + w_clash * l_clash + w_ca_ca * l_ca_ca)

    return {
        "total": total,
        "bond": l_bond,
        "angle": l_angle,
        "omega": l_omega,
        "rama": l_rama,
        "clash": l_clash,
        "ca_ca": l_ca_ca,
    }
