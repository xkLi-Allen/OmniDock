# -*- coding: utf-8 -*-
"""
Stage 4: Torsion-based Backbone Generation with Proper NeRF Geometry.

Key design:
  - Model predicts torsion angles (phi, psi, omega) as sin/cos pairs
  - Backbone coordinates built deterministically from torsions via NeRF
  - Strong backbone geometry priors during refinement
  - Pocket-conditioned generation: receptor surface guides ligand backbone
"""

import os
import sys
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import (
    BOND_N_CA, BOND_CA_C, BOND_C_N, BOND_C_O,
    ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_C_N_CA, ANGLE_CA_C_O,
    sincos_to_torsion, build_backbone_from_torsions, canonicalize_torsion_sincos,
    HELIX_PHI, HELIX_PSI, BETA_PHI, BETA_PSI,
)
from backbone_losses import combined_backbone_loss
from stage4_model import TorsionGenerator

COIL_PHI = -75.0
COIL_PSI = 145.0


def normalize_torsion_sincos(torsion_sincos: torch.Tensor) -> torch.Tensor:
    """Normalize each torsion's sin/cos pair independently.

    Representation is [sin_phi, sin_psi, sin_omega, cos_phi, cos_psi, cos_omega].
    """
    return canonicalize_torsion_sincos(torsion_sincos, assume_legacy_omega=False)


def angular_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


def make_ss_target(ss_mode: str, L: int, device: torch.device) -> torch.Tensor:
    """Return target SS labels: 0=helix, 1=beta, 2=coil."""
    if ss_mode == "helix":
        return torch.zeros(1, L, dtype=torch.long, device=device)
    if ss_mode == "beta":
        return torch.ones(1, L, dtype=torch.long, device=device)
    if ss_mode == "mixed":
        ss = torch.full((1, L), 2, dtype=torch.long, device=device)
        if L <= 12:
            ss[:, :] = 0
        else:
            h1 = max(4, L // 3)
            e1 = max(h1 + 3, 2 * L // 3)
            ss[:, :h1] = 0
            ss[:, h1:e1] = 2
            ss[:, e1:] = 1
        return ss
    return torch.full((1, L), 2, dtype=torch.long, device=device)


def ss_torsion_loss(torsion_sincos: torch.Tensor, ss_target: torch.Tensor) -> torch.Tensor:
    """Encourage continuous helix/beta torsion regions and trans omega."""
    torsion_sincos = normalize_torsion_sincos(torsion_sincos)
    torsions = sincos_to_torsion(torsion_sincos)
    phi = torsions[..., 0]
    psi = torsions[..., 1]
    omega = torsions[..., 2]

    helix_phi = torch.full_like(phi, HELIX_PHI)
    helix_psi = torch.full_like(psi, HELIX_PSI)
    beta_phi = torch.full_like(phi, BETA_PHI)
    beta_psi = torch.full_like(psi, BETA_PSI)
    trans_omega = torch.full_like(omega, 180.0)

    helix_mask = ss_target == 0
    beta_mask = ss_target == 1
    coil_mask = ss_target == 2

    loss = torsion_sincos.new_tensor(0.0)
    count = torsion_sincos.new_tensor(0.0)
    if helix_mask.any():
        loss = loss + (angular_diff_deg(phi[helix_mask], helix_phi[helix_mask]) ** 2).mean()
        loss = loss + (angular_diff_deg(psi[helix_mask], helix_psi[helix_mask]) ** 2).mean()
        count = count + 2.0
    if beta_mask.any():
        loss = loss + (angular_diff_deg(phi[beta_mask], beta_phi[beta_mask]) ** 2).mean()
        loss = loss + (angular_diff_deg(psi[beta_mask], beta_psi[beta_mask]) ** 2).mean()
        count = count + 2.0
    if coil_mask.any():
        coil_phi = torch.full_like(phi, COIL_PHI)
        coil_psi = torch.full_like(psi, COIL_PSI)
        loss = loss + 0.25 * (angular_diff_deg(phi[coil_mask], coil_phi[coil_mask]) ** 2).mean()
        loss = loss + 0.25 * (angular_diff_deg(psi[coil_mask], coil_psi[coil_mask]) ** 2).mean()
        count = count + 0.5

    loss = loss / count.clamp(min=1.0)
    omega_loss = (angular_diff_deg(omega, trans_omega) ** 2).mean()
    return (loss + 0.5 * omega_loss) / 1000.0


def ss_target_to_torsion_sincos(ss_target: torch.Tensor,
                                fallback: torch.Tensor = None) -> torch.Tensor:
    """Build target torsion sin/cos from SS labels.

    Representation is [sin_phi, sin_psi, sin_omega, cos_phi, cos_psi, cos_omega].
    Coil residues use fallback if provided, otherwise helix-like torsions.
    """
    B, L = ss_target.shape
    device = ss_target.device
    dtype = fallback.dtype if fallback is not None else torch.float32

    phi = torch.full((B, L), HELIX_PHI, dtype=dtype, device=device)
    psi = torch.full((B, L), HELIX_PSI, dtype=dtype, device=device)
    omega = torch.full((B, L), 180.0, dtype=dtype, device=device)

    beta_mask = ss_target == 1
    phi[beta_mask] = BETA_PHI
    psi[beta_mask] = BETA_PSI

    coil_mask = ss_target == 2
    phi[coil_mask] = COIL_PHI
    psi[coil_mask] = COIL_PSI
    if fallback is not None and coil_mask.any():
        fallback_torsion = sincos_to_torsion(normalize_torsion_sincos(fallback))
        omega[coil_mask] = fallback_torsion[..., 2][coil_mask]

    rad = torch.stack([phi, psi, omega], dim=-1) * (math.pi / 180.0)
    return torch.cat([torch.sin(rad), torch.cos(rad)], dim=-1)


def _smooth_ss_1d(labels: torch.Tensor, min_len: int = 3) -> torch.Tensor:
    out = labels.clone()
    L = int(labels.numel())
    start = 0
    while start < L:
        val = int(labels[start].item())
        end = start + 1
        while end < L and int(labels[end].item()) == val:
            end += 1
        if val in (0, 1) and end - start < min_len:
            out[start:end] = 2
        start = end
    return out


def smooth_ss_labels_tensor(labels: torch.Tensor, min_len: int = 3) -> torch.Tensor:
    if labels.dim() == 1:
        return _smooth_ss_1d(labels, min_len=min_len)
    return torch.stack([_smooth_ss_1d(labels[b], min_len=min_len) for b in range(labels.shape[0])], dim=0)


def blend_with_ss_torsion(torsion_sincos: torch.Tensor, ss_target: torch.Tensor,
                          strength: float) -> torch.Tensor:
    if strength <= 0:
        return normalize_torsion_sincos(torsion_sincos)
    target = ss_target_to_torsion_sincos(ss_target, fallback=torsion_sincos)
    blended = (1.0 - strength) * normalize_torsion_sincos(torsion_sincos) + strength * target
    return normalize_torsion_sincos(blended)


def load_compatible_state_dict(model: nn.Module, state: dict, label: str = "checkpoint"):
    current = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = len(state) - len(matched)
    model.load_state_dict(matched, strict=False)
    print(f"[Stage4] Loaded {label}: matched={len(matched)}, skipped_shape_mismatch={skipped}")


def generate_backbone(torsion_sincos: torch.Tensor) -> torch.Tensor:
    """
    Convert predicted torsion sin/cos to backbone coordinates using NeRF.
    torsion_sincos: (B, L, 6) sin/cos of [phi, psi, omega]
    Returns: (B, L, 3, 3) coordinates of [N, CA, C] per residue
    """
    B, L, _ = torsion_sincos.shape
    torsion_sincos = normalize_torsion_sincos(torsion_sincos)
    torsions_deg = sincos_to_torsion(torsion_sincos)

    backbones = []
    for b in range(B):
        phi = torsions_deg[b, :, 0]
        psi = torsions_deg[b, :, 1]
        omega = torsions_deg[b, :, 2]
        bb = build_backbone_from_torsions(phi, psi, omega)
        backbones.append(bb)

    return torch.stack(backbones, dim=0)


def pocket_contact_loss(backbone, pocket_center, pocket_radius,
                        target_contact_frac=0.3, softness=1.5):
    """Differentiable symmetric pocket-contact fraction loss."""
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    ca = backbone[:, :, 1, :]
    pc = pocket_center[:, :3]
    pr = pocket_radius

    dist_to_pocket = (ca - pc.unsqueeze(1)).norm(dim=-1)
    contact_prob = torch.sigmoid((pr.unsqueeze(1) + 5.0 - dist_to_pocket) / softness)
    contact_frac = contact_prob.mean(dim=1)

    target = torch.full_like(contact_frac, float(target_contact_frac))
    return F.mse_loss(contact_frac, target) * 1000.0


def centroid_alignment_loss(backbone, pocket_center):
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    gen_centroid = backbone[:, :, 1, :].mean(dim=1)
    return F.smooth_l1_loss(gen_centroid, pocket_center[:, :3])


def pocket_shell_loss(backbone, pocket_center, pocket_radius, softness=1.5):
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    ca = backbone[:, :, 1, :]
    dist = (ca - pocket_center[:, None, :3]).norm(dim=-1)
    shell_radius = pocket_radius.squeeze(-1) if pocket_radius.dim() == 2 else pocket_radius
    return F.smooth_l1_loss(dist.mean(dim=1), shell_radius + softness)


def centroid_to_pocket_dist(backbone, pocket_center):
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    gen_centroid = backbone[:, :, 1, :].mean(dim=1)
    return (gen_centroid - pocket_center[:, :3]).norm(dim=-1)


def soft_pocket_contact_fraction(backbone, pocket_center, pocket_radius, softness=1.5):
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    ca = backbone[:, :, 1, :]
    dist_to_pocket = (ca - pocket_center[:, None, :3]).norm(dim=-1)
    contact_prob = torch.sigmoid((pocket_radius.unsqueeze(1) + 5.0 - dist_to_pocket) / softness)
    return contact_prob.mean(dim=1)


def _skew_matrix(v: torch.Tensor) -> torch.Tensor:
    B = v.shape[0]
    K = torch.zeros(B, 3, 3, dtype=v.dtype, device=v.device)
    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]
    return K


def rodrigues_rotation(rot_vec: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) exponential map for row-vector coordinates."""
    B = rot_vec.shape[0]
    dtype, device = rot_vec.dtype, rot_vec.device
    K = _skew_matrix(rot_vec)
    I = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
    theta2 = (rot_vec ** 2).sum(dim=-1, keepdim=True).unsqueeze(-1)
    theta = torch.sqrt(theta2.clamp(min=1e-12))
    small = theta < 1e-4
    A = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    Bcoef = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp(min=1e-12))
    return I + A * K + Bcoef * (K @ K)


def apply_rigid_transform(backbone: torch.Tensor, rot_vec: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Apply trainable rotation+translation around the CA centroid."""
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    center = backbone[:, :, 1, :].mean(dim=1, keepdim=True)  # (B,1,3)
    R = rodrigues_rotation(rot_vec)  # (B,3,3)
    x = backbone - center[:, None, :, :]
    x_rot = torch.einsum('blij,bjk->blik', x, R.transpose(-1, -2))
    return x_rot + center[:, None, :, :] + trans[:, None, None, :]

def _carbonyl_oxygen_position(backbone: np.ndarray, i: int) -> np.ndarray:
    """Approximate backbone carbonyl O for residue i from N/CA/C atoms."""
    N = backbone[:, 0, :]
    CA = backbone[:, 1, :]
    C = backbone[:, 2, :]
    L = backbone.shape[0]

    v_ca = CA[i] - C[i]
    v_ca = v_ca / (np.linalg.norm(v_ca) + 1e-8)
    if i < L - 1:
        v_n = N[i + 1] - C[i]
    else:
        v_n = N[i] - C[i]
    v_n = v_n / (np.linalg.norm(v_n) + 1e-8)

    direction = -(v_ca + v_n)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return C[i] + BOND_C_O * direction


def _ss_segments(ss_labels: np.ndarray, label: int):
    segments = []
    start = None
    for idx, value in enumerate(ss_labels.tolist(), start=1):
        if value == label and start is None:
            start = idx
        elif value != label and start is not None:
            end = idx - 1
            if end - start + 1 >= 3:
                segments.append((start, end))
            start = None
    if start is not None:
        end = len(ss_labels)
        if end - start + 1 >= 3:
            segments.append((start, end))
    return segments



def read_pdb_chain_ca_coords(pdb_path: str, chains: str) -> np.ndarray:
    wanted = set(chains)
    coords = []
    with open(pdb_path, errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if line[21].strip() not in wanted:
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No CA atoms found for chains {chains} in {pdb_path}")
    return np.asarray(coords, dtype=np.float32)



def native_interface_centroid(pdb_path: str, receptor_chains: str, ligand_chains: str, contact_cutoff: float = 10.0) -> np.ndarray:
    rec = read_pdb_chain_ca_coords(pdb_path, receptor_chains)
    lig = read_pdb_chain_ca_coords(pdb_path, ligand_chains)
    d = np.linalg.norm(lig[:, None, :] - rec[None, :, :], axis=-1)
    mask = d.min(axis=1) <= contact_cutoff
    if mask.sum() < 3:
        mask = np.ones(len(lig), dtype=bool)
    return lig[mask].mean(axis=0).astype(np.float32)


def write_pdb(backbone: np.ndarray, output_path: str, chain_id: str = "A",
              residue_name: str = "ALA", ss_labels: np.ndarray = None):
    """Write backbone coordinates to a PDB file with optional HELIX/SHEET records."""
    L = backbone.shape[0]
    atom_idx = 1
    lines = ["HEADER    STAGE4 GENERATED BACKBONE"]

    if ss_labels is not None:
        ss_labels = np.asarray(ss_labels, dtype=np.int64).reshape(-1)
        if ss_labels.shape[0] == L:
            serial = 1
            for start, end in _ss_segments(ss_labels, 0):
                lines.append(
                    f"HELIX  {serial:3d} {serial:3d} {residue_name:>3s} {chain_id:1s}{start:4d}  "
                    f"{residue_name:>3s} {chain_id:1s}{end:4d}  1                                  {end-start+1:5d}"
                )
                serial += 1
            sheet_serial = 1
            for start, end in _ss_segments(ss_labels, 1):
                lines.append(
                    f"SHEET  {sheet_serial:3d} A {1:2d} {residue_name:>3s} {chain_id:1s}{start:4d}  "
                    f"{residue_name:>3s} {chain_id:1s}{end:4d}  0"
                )
                sheet_serial += 1

    for i in range(L):
        res_num = i + 1
        atom_coords = {
            "N": backbone[i, 0],
            "CA": backbone[i, 1],
            "C": backbone[i, 2],
            "O": _carbonyl_oxygen_position(backbone, i),
        }
        for atom_name in ["N", "CA", "C", "O"]:
            x, y, z = np.asarray(atom_coords[atom_name], dtype=np.float32).tolist()
            element = "C" if atom_name == "CA" else atom_name[0]
            line = (
                f"ATOM  {atom_idx:5d} {atom_name:>4s} {residue_name:>3s} {chain_id:1s}"
                f"{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
            lines.append(line)
            atom_idx += 1

    lines.append(f"TER   {atom_idx:5d}      {residue_name:>3s} {chain_id:1s}{L:4d}")
    lines.append("END")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def validate_backbone(backbone: np.ndarray) -> dict:
    """Validate generated backbone geometry and report statistics."""
    L = backbone.shape[0]
    N = backbone[:, 0, :]
    CA = backbone[:, 1, :]
    C = backbone[:, 2, :]

    d_n_ca = np.linalg.norm(CA - N, axis=-1)
    d_ca_c = np.linalg.norm(C - CA, axis=-1)
    d_c_n = np.linalg.norm(N[1:] - C[:-1], axis=-1) if L > 1 else np.array([])
    d_ca_ca = np.linalg.norm(CA[1:] - CA[:-1], axis=-1) if L > 1 else np.array([])

    def angle_3pts(a, b, c):
        v1 = a - b
        v2 = c - b
        cos_a = np.sum(v1 * v2, axis=-1) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    a_n_ca_c = angle_3pts(N, CA, C)
    a_ca_c_n = angle_3pts(CA[:-1], C[:-1], N[1:]) if L > 1 else np.array([])
    a_c_n_ca = angle_3pts(C[:-1], N[1:], CA[1:]) if L > 1 else np.array([])

    n_clashes = 0
    if L >= 4:
        ca_dmat = np.linalg.norm(CA[:, None, :] - CA[None, :, :], axis=-1)
        for i in range(L):
            for j in range(i + 3, L):
                if ca_dmat[i, j] < 2.8:
                    n_clashes += 1

    return {
        'n_ca_mean': float(d_n_ca.mean()), 'n_ca_std': float(d_n_ca.std()),
        'ca_c_mean': float(d_ca_c.mean()), 'ca_c_std': float(d_ca_c.std()),
        'c_n_mean': float(d_c_n.mean()) if len(d_c_n) > 0 else 0.0,
        'c_n_std': float(d_c_n.std()) if len(d_c_n) > 0 else 0.0,
        'ca_ca_mean': float(d_ca_ca.mean()) if len(d_ca_ca) > 0 else 0.0,
        'ca_ca_std': float(d_ca_ca.std()) if len(d_ca_ca) > 0 else 0.0,
        'n_ca_c_angle_mean': float(a_n_ca_c.mean()),
        'ca_c_n_angle_mean': float(a_ca_c_n.mean()) if len(a_ca_c_n) > 0 else 0.0,
        'c_n_ca_angle_mean': float(a_c_n_ca.mean()) if len(a_c_n_ca) > 0 else 0.0,
        'n_clashes': n_clashes,
        'n_residues': L,
    }


def backbone_rg_torch(backbone: torch.Tensor) -> torch.Tensor:
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    ca = backbone[:, :, 1, :]
    center = ca.mean(dim=1, keepdim=True)
    return torch.sqrt(((ca - center) ** 2).sum(dim=-1).mean(dim=1) + 1e-8)


def ca_dist_preservation_loss(backbone: torch.Tensor, raw_backbone: torch.Tensor) -> torch.Tensor:
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    if raw_backbone.dim() == 3:
        raw_backbone = raw_backbone.unsqueeze(0)
    ca = backbone[:, :, 1, :]
    raw_ca = raw_backbone[:, :, 1, :]
    L = ca.shape[1]
    valid = ~torch.eye(L, dtype=torch.bool, device=ca.device).unsqueeze(0)
    with torch.amp.autocast("cuda", enabled=False):
        d = torch.cdist(ca.float(), ca.float())
        d0 = torch.cdist(raw_ca.float(), raw_ca.float())
        return F.smooth_l1_loss(d[valid], d0[valid]).to(ca.dtype)


def torsion_deviation_loss(torsion_sincos: torch.Tensor, raw_torsion_sincos: torch.Tensor) -> torch.Tensor:
    torsion = sincos_to_torsion(normalize_torsion_sincos(torsion_sincos))
    raw = sincos_to_torsion(normalize_torsion_sincos(raw_torsion_sincos))
    diff = angular_diff_deg(torsion, raw)
    return (diff ** 2).mean() / 1000.0


def rg_preservation_loss(backbone: torch.Tensor, raw_backbone: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(backbone_rg_torch(backbone), backbone_rg_torch(raw_backbone))


def refinement_proxy_score(backbone: torch.Tensor, raw_backbone: torch.Tensor,
                           pocket_center: torch.Tensor, pocket_radius: torch.Tensor,
                           softness: float, target_contact_frac: float,
                           max_rg_ratio: float) -> tuple[float, dict]:
    with torch.no_grad():
        contact_frac = soft_pocket_contact_fraction(backbone, pocket_center, pocket_radius, softness).mean()
        rg = backbone_rg_torch(backbone).mean()
        raw_rg = backbone_rg_torch(raw_backbone).mean().clamp(min=1e-6)
        rg_ratio = rg / raw_rg
        rg_penalty = (rg_ratio - 1.0).abs() + (rg_ratio - max_rg_ratio).clamp_min(0.0) * 5.0
        ca_drift = ca_dist_preservation_loss(backbone, raw_backbone)
        center_dist = centroid_to_pocket_dist(backbone, pocket_center).mean()
        clash_report = validate_backbone(backbone[0].detach().cpu().numpy())
        clash_penalty = torch.tensor(float(clash_report['n_clashes']) / 20.0, device=backbone.device, dtype=backbone.dtype)
        contact_score = 1.0 - (target_contact_frac - contact_frac).abs().clamp(max=1.0)
        center_penalty = center_dist / 10.0
        proxy = contact_score - center_penalty - clash_penalty - rg_penalty - ca_drift
        metrics = {
            'contact_frac': float(contact_frac.item()),
            'center_dist': float(center_dist.item()),
            'rg': float(rg.item()),
            'raw_rg': float(raw_rg.item()),
            'rg_ratio': float(rg_ratio.item()),
            'ca_drift': float(ca_drift.item()),
            'clashes': int(clash_report['n_clashes']),
            'proxy': float(proxy.item()),
        }
        return float(proxy.item()), metrics


def main():
    ap = argparse.ArgumentParser(description="Stage 4: Torsion-based Backbone Generation")
    ap.add_argument("--rec_npz", type=str, required=True,
                    help="Receptor .npz from Stage 1")
    ap.add_argument("--stage3_ckpt", type=str, default="",
                    help="Stage 3 checkpoint for encoder init")
    ap.add_argument("--stage4_ckpt", type=str, default="",
                    help="Stage 4 trained generator checkpoint")
    ap.add_argument("--num_residues", type=int, default=30)
    ap.add_argument("--output_pdb", type=str, default="generated_backbone.pdb")
    ap.add_argument("--refine_steps", type=int, default=100)
    ap.add_argument("--refinement_mode", type=str, default="rigid_only",
                    choices=["raw", "rigid_only", "constrained_torsion", "legacy_torsion"],
                    help="Generation refinement mode. Default keeps torsions frozen and only refines rigid placement.")
    ap.add_argument("--enable_torsion_refine", action="store_true",
                    help="Shortcut for --refinement_mode legacy_torsion; required for old strong torsion refinement behavior")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers_surf", type=int, default=6)
    ap.add_argument("--nlayers_gen", type=int, default=4)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--w_bond", type=float, default=100.0)
    ap.add_argument("--w_angle", type=float, default=50.0)
    ap.add_argument("--w_omega", type=float, default=20.0)
    ap.add_argument("--w_rama", type=float, default=10.0)
    ap.add_argument("--w_clash", type=float, default=5.0)
    ap.add_argument("--w_pocket", type=float, default=2.0,
                    help="Legacy pocket/contact refinement weight used by torsion refinement modes")
    ap.add_argument("--w_center_refine", type=float, default=1.0,
                    help="Weight for generated centroid to predicted pocket-center alignment in refinement")
    ap.add_argument("--w_contact_refine", type=float, default=0.01,
                    help="Weight for symmetric soft pocket-contact fraction loss in refinement")
    ap.add_argument("--w_shell_refine", type=float, default=0.0,
                    help="Optional weight for matching mean CA distance to the predicted pocket radius shell")
    ap.add_argument("--w_ss_torsion", type=float, default=0.0,
                    help="Weight for explicit helix/beta torsion constraints during refinement")
    ap.add_argument("--ss_mode", type=str, default="free",
                    choices=["free", "helix", "beta", "mixed"],
                    help="Optional forced secondary-structure mode for Stage4 refinement")
    ap.add_argument("--ss_source", type=str, default="none", choices=["none", "model_smooth", "forced"],
                    help="Source for generation-time weak SS torsion constraints. Default avoids noisy model argmax.")
    ap.add_argument("--ss_header_source", type=str, default="model_smooth", choices=["none", "model_smooth", "refinement"],
                    help="Annotation-only source for HELIX/SHEET PDB records; does not affect torsions or refinement.")
    ap.add_argument("--placement_center_source", type=str, default="predicted", choices=["predicted", "native_centroid", "native_interface_centroid", "xyz"],
                    help="Placement center source. native_centroid/xyz are diagnostic oracle modes, not deployable generation defaults.")
    ap.add_argument("--native_pdb", type=str, default="", help="Native PDB for --placement_center_source native_centroid")
    ap.add_argument("--ligand_chains", type=str, default="", help="Native ligand chains for oracle placement center modes")
    ap.add_argument("--receptor_chains", type=str, default="", help="Native receptor chains for --placement_center_source native_interface_centroid")
    ap.add_argument("--native_contact_cutoff", type=float, default=10.0, help="Contact cutoff for native_interface_centroid oracle mode")
    ap.add_argument("--placement_center_xyz", type=float, nargs=3, default=None, help="Explicit placement center for --placement_center_source xyz")
    ap.add_argument("--force_accept_refined", action="store_true", help="Diagnostic: save final refined backbone even if proxy prefers raw")
    ap.add_argument("--residue_name", type=str, default="ALA",
                    help="Residue name written to backbone-only PDB; sequence design is not performed here")
    ap.add_argument("--ss_init_strength", type=float, default=0.0,
                    help="Blend initial torsions toward ss_mode targets before refinement; use 1.0 for strong helix/beta initialization")
    ap.add_argument("--refine_lr", type=float, default=0.01,
                    help="Rigid-body refinement learning rate")
    ap.add_argument("--torsion_refine_lr", type=float, default=1e-4,
                    help="Small torsion learning rate for constrained_torsion mode")
    ap.add_argument("--target_contact_frac", type=float, default=0.30)
    ap.add_argument("--pocket_softness", type=float, default=1.5)
    ap.add_argument("--disable_rigid", action="store_true",
                    help="Disable rotation/translation refinement; not recommended")
    ap.add_argument("--save_raw_pdb", action="store_true",
                    help="Save raw model backbone before refinement")
    ap.add_argument("--raw_output_pdb", type=str, default="",
                    help="Path for raw model backbone PDB; defaults to <output>_raw.pdb")
    ap.add_argument("--freeze_torsion_refine", action="store_true",
                    help="Deprecated alias for --refinement_mode rigid_only")
    ap.add_argument("--w_torsion_deviation", type=float, default=100.0)
    ap.add_argument("--w_ca_dist_preserve", type=float, default=50.0)
    ap.add_argument("--w_rg_preserve", type=float, default=50.0)
    ap.add_argument("--max_refine_rg_ratio", type=float, default=1.5)
    ap.add_argument("--best_check_every", type=int, default=5)
    args = ap.parse_args()

    if args.enable_torsion_refine:
        args.refinement_mode = "legacy_torsion"
    if args.freeze_torsion_refine:
        args.refinement_mode = "rigid_only"

    device = torch.device(args.device)

    # Load Stage4 checkpoint first, then use its hyperparameters. This avoids
    # silent partial loading when K/d_model/layers differ between train and generate.
    stage4_ckpt_obj = None
    generator_max_residues = args.num_residues
    if args.stage4_ckpt and os.path.isfile(args.stage4_ckpt):
        stage4_ckpt_obj = torch.load(args.stage4_ckpt, map_location=device)
        ckpt_args = stage4_ckpt_obj.get("args", {})
        args.d_model = int(ckpt_args.get("d_model", args.d_model))
        args.nhead = int(ckpt_args.get("nhead", args.nhead))
        args.nlayers_surf = int(ckpt_args.get("nlayers_surf", args.nlayers_surf))
        args.nlayers_gen = int(ckpt_args.get("nlayers_gen", args.nlayers_gen))
        args.K = int(ckpt_args.get("K", args.K))
        generator_max_residues = int(ckpt_args.get("max_residues", args.num_residues))
        generator_max_residues = max(generator_max_residues, args.num_residues)
        print(f"[Stage4] Using checkpoint hyperparameters: d_model={args.d_model}, "
              f"nhead={args.nhead}, K={args.K}, max_residues={generator_max_residues}")

    # Load receptor surface after K has possibly been overwritten by ckpt args.
    print(f"[Stage4] Loading receptor surface from {args.rec_npz}")
    with np.load(args.rec_npz, allow_pickle=True) as data:
        xs = data['xs'].astype(np.float32)
        ns = data['ns'].astype(np.float32)
        patch_centers = data['patch_centers'].astype(np.float32)
        patch_knn_idx = data['patch_knn_idx'].astype(np.int64)
        patch_order = data['patch_order'].astype(np.int64)

    K0 = patch_knn_idx.shape[1]
    if K0 < args.K:
        patch_knn_idx = np.concatenate(
            [patch_knn_idx, np.tile(patch_knn_idx[:, -1:], (1, args.K - K0))], axis=1)
    elif K0 > args.K:
        patch_knn_idx = patch_knn_idx[:, :args.K]

    Nc = patch_centers.shape[0]
    sel = patch_order[:min(Nc, args.seq_len)]

    pts_idx = patch_knn_idx[sel]
    ctrs = patch_centers[sel]
    rel_xyz = xs[pts_idx] - ctrs[:, None, :]
    norms = ns[pts_idx]
    feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)

    rec_feats = torch.from_numpy(feats).unsqueeze(0).to(device)
    rec_centers = torch.from_numpy(ctrs).unsqueeze(0).to(device)
    rec_pad = torch.zeros(1, rec_feats.shape[1], dtype=torch.bool, device=device)

    # Build generator
    print("[Stage4] Building generator model...")
    generator = TorsionGenerator(
        d_model=args.d_model, nhead=args.nhead,
        nlayers_surf=args.nlayers_surf, nlayers_gen=args.nlayers_gen,
        K=args.K, max_residues=generator_max_residues,
    ).to(device)

    if args.stage3_ckpt:
        generator.load_from_stage3(args.stage3_ckpt, device=device)

    if stage4_ckpt_obj is not None:
        load_compatible_state_dict(generator, stage4_ckpt_obj["model"], label=args.stage4_ckpt)

    # Initial torsion prediction
    print(f"[Stage4] Generating {args.num_residues} residues...")
    generator.eval()
    with torch.no_grad():
        torsion_sincos, ss_logits, pocket_pred, _hotspot_logits, _topology_logits = generator(
            rec_feats, rec_centers, rec_pad, num_residues=args.num_residues
        )
        torsion_sincos = normalize_torsion_sincos(torsion_sincos)

    ss_target = None
    if args.ss_source == "forced":
        if args.ss_mode == "free":
            raise ValueError("--ss_source forced requires --ss_mode helix/beta/mixed")
        ss_target = make_ss_target(args.ss_mode, args.num_residues, device)
        if args.ss_init_strength > 0:
            torsion_sincos = blend_with_ss_torsion(torsion_sincos, ss_target, args.ss_init_strength)
            print(f"[Stage4] Blended initial torsions toward {args.ss_mode}: strength={args.ss_init_strength}")
        print(f"[Stage4] Using forced SS constraint: {args.ss_mode}")
    elif args.ss_source == "model_smooth":
        ss_target = smooth_ss_labels_tensor(ss_logits.argmax(dim=-1).detach(), min_len=3)
        print("[Stage4] Using smoothed model-predicted SS as weak refinement target")
    elif args.ss_source == "none":
        if args.w_ss_torsion > 0:
            print("[Stage4][WARN] --w_ss_torsion ignored because --ss_source=none")
            args.w_ss_torsion = 0.0
    else:
        raise ValueError(f"Unknown ss_source: {args.ss_source}")

    ss_header_labels = None
    if args.ss_header_source == "model_smooth":
        ss_header_labels = smooth_ss_labels_tensor(ss_logits.argmax(dim=-1).detach(), min_len=3)
        h = (ss_header_labels == 0).float().mean().item()
        e = (ss_header_labels == 1).float().mean().item()
        c = (ss_header_labels == 2).float().mean().item()
        print(f"[Stage4] Writing annotation-only SS headers from smoothed model predictions: H={h:.3f} E={e:.3f} C={c:.3f}")
    elif args.ss_header_source == "refinement":
        ss_header_labels = ss_target
        if ss_header_labels is None:
            print("[Stage4][WARN] --ss_header_source=refinement requested but no refinement SS target is active; no SS headers will be written")
    elif args.ss_header_source != "none":
        raise ValueError(f"Unknown ss_header_source: {args.ss_header_source}")

    # Refinement with geometry constraints
    print(f"[Stage4] Refinement mode: {args.refinement_mode}; steps={args.refine_steps}")
    torsion_param = nn.Parameter(torsion_sincos.detach().clone())

    valid_rec = ~rec_pad
    rec_center_ref = (rec_centers * valid_rec.unsqueeze(-1).float()).sum(dim=1) / valid_rec.float().sum(dim=1, keepdim=True).clamp(min=1)
    predicted_pocket_center = (rec_center_ref + pocket_pred[:, :3]).detach()
    pocket_center = predicted_pocket_center
    pocket_radius = (F.softplus(pocket_pred[:, 3:4]) + 1e-6).detach()
    if args.placement_center_source == "native_centroid":
        if not args.native_pdb or not args.ligand_chains:
            raise ValueError("--placement_center_source native_centroid requires --native_pdb and --ligand_chains")
        native_ca = read_pdb_chain_ca_coords(args.native_pdb, args.ligand_chains)
        native_center_np = native_ca.mean(axis=0)
        pocket_center = torch.as_tensor(native_center_np, dtype=predicted_pocket_center.dtype, device=device).view(1, 3)
        print(f"[Stage4][ORACLE] Overriding placement center with native ligand centroid: {native_center_np.tolist()}")
    elif args.placement_center_source == "native_interface_centroid":
        if not args.native_pdb or not args.receptor_chains or not args.ligand_chains:
            raise ValueError("--placement_center_source native_interface_centroid requires --native_pdb, --receptor_chains, and --ligand_chains")
        native_center_np = native_interface_centroid(args.native_pdb, args.receptor_chains, args.ligand_chains, args.native_contact_cutoff)
        pocket_center = torch.as_tensor(native_center_np, dtype=predicted_pocket_center.dtype, device=device).view(1, 3)
        print(f"[Stage4][ORACLE] Overriding placement center with native interface centroid: {native_center_np.tolist()}")
    elif args.placement_center_source == "xyz":
        if args.placement_center_xyz is None:
            raise ValueError("--placement_center_source xyz requires --placement_center_xyz X Y Z")
        pocket_center = torch.tensor(args.placement_center_xyz, dtype=predicted_pocket_center.dtype, device=device).view(1, 3)
        print(f"[Stage4][ORACLE] Overriding placement center with xyz: {args.placement_center_xyz}")
    pred_center_np = predicted_pocket_center[0].detach().cpu().numpy().tolist()
    used_center_np = pocket_center[0].detach().cpu().numpy().tolist()
    print(f"[Stage4] Predicted pocket center: ({pred_center_np[0]:.3f}, {pred_center_np[1]:.3f}, {pred_center_np[2]:.3f})")
    print(f"[Stage4] Used placement center: ({used_center_np[0]:.3f}, {used_center_np[1]:.3f}, {used_center_np[2]:.3f})")

    # Initialize generated backbone at the predicted pocket. Torsions control the
    # internal shape; rigid variables control docking placement.
    with torch.no_grad():
        init_bb = generate_backbone(torsion_sincos)
        init_ca_center = init_bb[:, :, 1, :].mean(dim=1)  # (B, 3)
        init_trans = (pocket_center - init_ca_center).to(torsion_sincos.dtype)  # (B, 3)
        raw_bb = init_bb + init_trans[:, None, None, :]

    if args.save_raw_pdb:
        raw_output = args.raw_output_pdb
        if not raw_output:
            root, ext = os.path.splitext(args.output_pdb)
            raw_output = f"{root}_raw{ext or '.pdb'}"
        raw_np = raw_bb[0].detach().cpu().numpy()
        if raw_np.ndim == 4:
            raw_np = raw_np[0]
        raw_report = validate_backbone(raw_np)
        print("\n[Stage4] Raw backbone validation:")
        print(f"  CA-CA distance: {raw_report['ca_ca_mean']:.3f} +/- {raw_report['ca_ca_std']:.3f} A")
        print(f"  Clashes: {raw_report['n_clashes']}")
        raw_ss_np = ss_header_labels[0].detach().cpu().numpy() if ss_header_labels is not None else None
        write_pdb(raw_np, raw_output, residue_name=args.residue_name, ss_labels=raw_ss_np)
        print(f"[Stage4] Raw PDB written to {raw_output}")

    rot_vec = nn.Parameter(torch.zeros(init_trans.shape[0], 3, device=device, dtype=torsion_sincos.dtype))
    trans = nn.Parameter(init_trans.detach().clone())

    raw_proxy, raw_proxy_metrics = refinement_proxy_score(
        raw_bb, raw_bb, pocket_center, pocket_radius,
        args.pocket_softness, args.target_contact_frac, args.max_refine_rg_ratio)
    best_proxy = raw_proxy
    best_backbone = raw_bb.detach().clone()
    best_step = 0
    best_metrics = raw_proxy_metrics
    last_backbone = raw_bb.detach().clone()

    if args.refinement_mode == "raw" or args.refine_steps <= 0:
        print("[Stage4] Raw mode: skipping refinement and saving raw backbone")
    else:
        if args.refinement_mode == "rigid_only":
            opt = torch.optim.Adam([rot_vec, trans], lr=args.refine_lr)
            print("[Stage4] Refinement mode: rigid_only; torsions frozen")
        elif args.refinement_mode == "constrained_torsion":
            opt = torch.optim.Adam([
                {"params": [rot_vec, trans], "lr": args.refine_lr},
                {"params": [torsion_param], "lr": args.torsion_refine_lr},
            ])
            print(f"[Stage4] Refinement mode: constrained_torsion; rigid_lr={args.refine_lr}, torsion_lr={args.torsion_refine_lr}")
        elif args.refinement_mode == "legacy_torsion":
            params = [torsion_param] if args.disable_rigid else [torsion_param, rot_vec, trans]
            opt = torch.optim.Adam(params, lr=args.refine_lr)
            print("[Stage4] Refinement mode: legacy_torsion; old strong torsion+rigid behavior")
        else:
            raise ValueError(f"Unknown refinement_mode: {args.refinement_mode}")

        for step in range(args.refine_steps):
            opt.zero_grad()

            torsion_normalized = normalize_torsion_sincos(torsion_param)
            backbone_internal = generate_backbone(torsion_normalized)
            backbone = backbone_internal if args.disable_rigid else apply_rigid_transform(backbone_internal, rot_vec, trans)
            last_backbone = backbone.detach().clone()

            losses = combined_backbone_loss(
                backbone_internal, ss_target=None,
                w_bond=args.w_bond, w_angle=args.w_angle,
                w_omega=args.w_omega, w_rama=args.w_rama, w_clash=args.w_clash,
                w_ca_ca=100.0,
            )
            l_pocket_contact = pocket_contact_loss(
                backbone, pocket_center, pocket_radius,
                target_contact_frac=args.target_contact_frac, softness=args.pocket_softness)
            l_center = centroid_alignment_loss(backbone, pocket_center)
            l_shell = pocket_shell_loss(backbone, pocket_center, pocket_radius, softness=args.pocket_softness)
            l_ss_torsion = torsion_param.new_tensor(0.0)
            if ss_target is not None and args.w_ss_torsion > 0:
                l_ss_torsion = ss_torsion_loss(torsion_normalized, ss_target)

            l_torsion_dev = torsion_param.new_tensor(0.0)
            l_ca_preserve = torsion_param.new_tensor(0.0)
            l_rg_preserve = torsion_param.new_tensor(0.0)
            if args.refinement_mode == "constrained_torsion":
                l_torsion_dev = torsion_deviation_loss(torsion_normalized, torsion_sincos.detach())
                l_ca_preserve = ca_dist_preservation_loss(backbone_internal, init_bb.detach())
                l_rg_preserve = rg_preservation_loss(backbone_internal, init_bb.detach())
                total = (
                    losses['total'] + args.w_center_refine * l_center +
                    args.w_contact_refine * l_pocket_contact + args.w_shell_refine * l_shell +
                    args.w_ss_torsion * l_ss_torsion +
                    args.w_torsion_deviation * l_torsion_dev +
                    args.w_ca_dist_preserve * l_ca_preserve +
                    args.w_rg_preserve * l_rg_preserve
                )
            elif args.refinement_mode == "rigid_only":
                total = args.w_center_refine * l_center + args.w_contact_refine * l_pocket_contact + args.w_shell_refine * l_shell
            else:
                total = losses['total'] + args.w_center_refine * l_center + args.w_contact_refine * l_pocket_contact + args.w_shell_refine * l_shell + args.w_ss_torsion * l_ss_torsion

            total.backward()
            opt.step()

            if (step + 1) % max(1, args.best_check_every) == 0 or (step + 1) == args.refine_steps:
                proxy, proxy_metrics = refinement_proxy_score(
                    backbone.detach(), raw_bb.detach(), pocket_center, pocket_radius,
                    args.pocket_softness, args.target_contact_frac, args.max_refine_rg_ratio)
                if proxy > best_proxy:
                    best_proxy = proxy
                    best_backbone = backbone.detach().clone()
                    best_step = step + 1
                    best_metrics = proxy_metrics

            if (step + 1) % 20 == 0:
                contact_frac = soft_pocket_contact_fraction(backbone.detach(), pocket_center, pocket_radius, args.pocket_softness).mean().item()
                current_rg = backbone_rg_torch(backbone.detach()).mean().item()
                center_dist = centroid_to_pocket_dist(backbone.detach(), pocket_center).mean().item()
                centroid = backbone.detach()[:, :, 1, :].mean(dim=1)[0].detach().cpu().numpy().tolist()
                print(f"  Step {step+1}: total={total.item():.4f} "
                      f"center_loss={l_center.item():.4f} "
                      f"contact_loss={l_pocket_contact.item():.4f} "
                      f"shell_loss={l_shell.item():.4f} "
                      f"geometry={losses['total'].item():.4f} "
                      f"clash={losses['clash'].item():.4f} "
                      f"torsion_dev={l_torsion_dev.item():.4f} "
                      f"ca_preserve={l_ca_preserve.item():.4f} "
                      f"rg_preserve={l_rg_preserve.item():.4f} "
                      f"current_rg={current_rg:.3f} "
                      f"contact_frac={contact_frac:.3f} "
                      f"centroid_to_pocket_dist={center_dist:.3f} "
                      f"centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}) "
                      f"best_proxy={best_proxy:.4f}@{best_step}")

        print(f"[Stage4] Best refinement proxy={best_proxy:.4f} at step={best_step}; raw_proxy={raw_proxy:.4f}; metrics={best_metrics}")
        if args.force_accept_refined:
            print("[Stage4][DIAGNOSTIC] force_accept_refined enabled; saving final refined backbone")
            best_backbone = last_backbone
            best_step = args.refine_steps
        elif best_proxy <= raw_proxy:
            print("[Stage4] Best refined proxy did not improve over raw; reverting to raw backbone")
            best_backbone = raw_bb.detach().clone()
            best_step = 0

    backbone_final = best_backbone

    bb_np = backbone_final[0].detach().cpu().numpy()
    if bb_np.ndim == 4:
        bb_np = bb_np[0]

    # Validation report
    report = validate_backbone(bb_np)
    print(f"\n[Stage4] Backbone validation:")
    print(f"  N-CA bond: {report['n_ca_mean']:.3f} +/- {report['n_ca_std']:.3f} A "
          f"(ideal: {BOND_N_CA:.3f})")
    print(f"  CA-C bond: {report['ca_c_mean']:.3f} +/- {report['ca_c_std']:.3f} A "
          f"(ideal: {BOND_CA_C:.3f})")
    print(f"  C-N bond:  {report['c_n_mean']:.3f} +/- {report['c_n_std']:.3f} A "
          f"(ideal: {BOND_C_N:.3f})")
    print(f"  CA-CA distance: {report['ca_ca_mean']:.3f} +/- {report['ca_ca_std']:.3f} A "
          f"(expected: ~3.8)")
    print(f"  N-CA-C angle: {report['n_ca_c_angle_mean']:.1f} deg "
          f"(ideal: {ANGLE_N_CA_C:.1f})")
    print(f"  CA-C-N angle: {report['ca_c_n_angle_mean']:.1f} deg "
          f"(ideal: {ANGLE_CA_C_N:.1f})")
    print(f"  C-N-CA angle: {report['c_n_ca_angle_mean']:.1f} deg "
          f"(ideal: {ANGLE_C_N_CA:.1f})")
    print(f"  Clashes: {report['n_clashes']}")
    print(f"  Residues: {report['n_residues']}")

    ss_np = ss_header_labels[0].detach().cpu().numpy() if ss_header_labels is not None else None
    write_pdb(bb_np, args.output_pdb, residue_name=args.residue_name, ss_labels=ss_np)
    print(f"\n[Stage4] PDB written to {args.output_pdb}")


if __name__ == "__main__":
    main()
