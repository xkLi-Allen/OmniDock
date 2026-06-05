# -*- coding: utf-8 -*-

"""

Core geometry utilities for protein backbone manipulation.

Includes: NeRF coordinate placement, torsion angle computation,

ideal backbone geometry constants, and Ramachandran validation.

"""



import math

import numpy as np

import torch

import torch.nn.functional as F





# ============================================================

# Ideal backbone geometry constants

# ============================================================



BOND_N_CA = 1.458   # Å

BOND_CA_C = 1.525   # Å

BOND_C_N = 1.329    # Å (peptide bond)

BOND_C_O = 1.231    # Å



ANGLE_N_CA_C = 111.2    # degrees

ANGLE_CA_C_N = 116.2    # degrees

ANGLE_C_N_CA = 121.7    # degrees

ANGLE_CA_C_O = 120.8    # degrees



OMEGA_TRANS = 180.0  # peptide dihedral for trans in the sin/cos torsion convention



HELIX_PHI = -57.0

HELIX_PSI = -47.0

BETA_PHI = -135.0

BETA_PSI = 135.0



AMINO20 = [

    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",

    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"

]

RES2IDX = {r: i for i, r in enumerate(AMINO20)}

IDX2RES = {i: r for i, r in enumerate(AMINO20)}



SS_LABELS = {"H": 0, "E": 1, "C": 2}  # Helix, Sheet, Coil

SS_NAMES = {0: "H", 1: "E", 2: "C"}





# ============================================================

# NeRF: Natural Extension Reference Frame

# ============================================================



def place_atom_nerf(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,

                    length: float, angle_deg: float, dihedral_deg: float) -> torch.Tensor:

    """

    Place atom D given atoms A, B, C using the NeRF algorithm.

    D is placed such that: |C-D| = length, angle(B,C,D) = angle_deg, dihedral(A,B,C,D) = dihedral_deg

    """

    dtype = c.dtype

    device = c.device



    theta = torch.tensor(angle_deg * math.pi / 180.0, dtype=dtype, device=device)

    chi = torch.tensor(dihedral_deg * math.pi / 180.0, dtype=dtype, device=device)



    bc = c - b

    bc_norm = bc / bc.norm().clamp(min=1e-8)



    ba = a - b

    n = torch.cross(ba, bc, dim=0)

    n_len = n.norm()

    if n_len < 1e-8:

        perp = torch.zeros(3, dtype=dtype, device=device)

        idx = bc_norm.abs().argmin()

        perp[idx] = 1.0

        n = torch.cross(bc_norm, perp, dim=0)

    n = n / n.norm().clamp(min=1e-8)



    m = torch.cross(n, bc_norm, dim=0)

    m = m / m.norm().clamp(min=1e-8)



    d_dir = (

        -torch.cos(theta) * bc_norm

        + torch.sin(theta) * torch.cos(chi) * m

        + torch.sin(theta) * torch.sin(chi) * n

    )



    return c + length * d_dir





def place_atom_nerf_batch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,

                          length: torch.Tensor, angle_deg: torch.Tensor,

                          dihedral_deg: torch.Tensor) -> torch.Tensor:

    """Batched NeRF placement. All inputs are (B, 3) or (B,) tensors."""

    theta = angle_deg * (math.pi / 180.0)

    chi = dihedral_deg * (math.pi / 180.0)



    bc = c - b

    bc_norm = bc / bc.norm(dim=-1, keepdim=True).clamp(min=1e-8)



    ba = a - b

    n = torch.cross(ba, bc, dim=-1)

    n_len = n.norm(dim=-1, keepdim=True)

    degenerate = (n_len < 1e-8).squeeze(-1)

    if degenerate.any():

        perp = torch.zeros_like(bc_norm)

        idx = bc_norm[degenerate].abs().argmin(dim=-1)

        for i, di in enumerate(degenerate.nonzero(as_tuple=False).squeeze(-1)):

            perp[di, idx[i]] = 1.0

        n[degenerate] = torch.cross(bc_norm[degenerate], perp[degenerate], dim=-1)

    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)



    m = torch.cross(n, bc_norm, dim=-1)

    m = m / m.norm(dim=-1, keepdim=True).clamp(min=1e-8)



    cos_theta = torch.cos(theta).unsqueeze(-1)

    sin_theta = torch.sin(theta).unsqueeze(-1)

    cos_chi = torch.cos(chi).unsqueeze(-1)

    sin_chi = torch.sin(chi).unsqueeze(-1)



    d_dir = -cos_theta * bc_norm + sin_theta * cos_chi * m + sin_theta * sin_chi * n

    return c + length.unsqueeze(-1) * d_dir





# ============================================================

# Torsion angle computation

# ============================================================



def compute_dihedral(p0: np.ndarray, p1: np.ndarray,

                     p2: np.ndarray, p3: np.ndarray) -> float:

    """Compute dihedral angle (in degrees) defined by four points."""

    b0 = p1 - p0

    b1 = p2 - p1

    b2 = p3 - p2



    b1_norm = b1 / (np.linalg.norm(b1) + 1e-8)



    v = b0 - np.dot(b0, b1_norm) * b1_norm

    w = b2 - np.dot(b2, b1_norm) * b1_norm



    x = np.dot(v, w)

    y = np.dot(np.cross(b1_norm, v), w)



    return math.degrees(math.atan2(y, x))





def compute_dihedral_torch(p0: torch.Tensor, p1: torch.Tensor,

                           p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:

    """Compute dihedral angle (in degrees) for batched inputs. Each input: (..., 3)."""

    b0 = p1 - p0

    b1 = p2 - p1

    b2 = p3 - p2



    b1_norm = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-8)



    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm

    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm



    x = (v * w).sum(dim=-1)

    y = (torch.cross(b1_norm, v, dim=-1) * w).sum(dim=-1)



    return torch.atan2(y, x) * (180.0 / math.pi)





def compute_backbone_torsions(backbone_ncac: np.ndarray,

                              valid_mask: np.ndarray) -> np.ndarray:

    """

    Compute phi, psi, omega from backbone N, CA, C coordinates.

    backbone_ncac: (L, 3, 3) where [:, 0, :] = N, [:, 1, :] = CA, [:, 2, :] = C

    Returns: (L, 3) array [phi, psi, omega] in degrees.

    """

    L = backbone_ncac.shape[0]

    torsions = np.zeros((L, 3), dtype=np.float32)



    N = backbone_ncac[:, 0, :]

    CA = backbone_ncac[:, 1, :]

    C = backbone_ncac[:, 2, :]



    for i in range(L):

        if not valid_mask[i]:

            continue

        if i > 0 and valid_mask[i - 1]:

            torsions[i, 0] = compute_dihedral(C[i - 1], N[i], CA[i], C[i])

        if i < L - 1 and valid_mask[i + 1]:

            torsions[i, 1] = compute_dihedral(N[i], CA[i], C[i], N[i + 1])

        if i < L - 1 and valid_mask[i + 1]:

            torsions[i, 2] = compute_dihedral(CA[i], C[i], N[i + 1], CA[i + 1])



    return torsions





def torsion_to_sincos(torsions_deg: np.ndarray) -> np.ndarray:

    """Convert (L, 3) torsion angles in degrees to (L, 6) sin/cos representation.

    Stage 4 convention is [phi, psi, omega] with trans omega at +/-180 degrees
    and cis omega at 0 degrees.
    """

    rad = torsions_deg * (math.pi / 180.0)

    sin_vals = np.sin(rad)

    cos_vals = np.cos(rad)

    return np.concatenate([sin_vals, cos_vals], axis=-1).astype(np.float32)



def wrap_angle_deg_torch(angle: torch.Tensor) -> torch.Tensor:

    """Wrap angles in degrees to [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0



def legacy_omega_to_trans180_torch(omega: torch.Tensor) -> torch.Tensor:

    """Convert legacy omega phase (trans=0) to Stage 4 convention (trans=180)."""

    return wrap_angle_deg_torch(omega + 180.0)



def canonicalize_torsion_sincos(torsion_sincos: torch.Tensor,
                                assume_legacy_omega: bool = True) -> torch.Tensor:

    """Normalize torsion sin/cos and optionally convert legacy omega phase.

    Representation is [sin_phi, sin_psi, sin_omega, cos_phi, cos_psi, cos_omega].
    Legacy NPZ files in this project store omega with trans peptide bonds near
    0 degrees. Stage 4 uses trans omega at +/-180 degrees, so adding 180 degrees
    to omega gives the canonical convention while leaving phi/psi unchanged.
    """

    sin_vals = torsion_sincos[..., :3]

    cos_vals = torsion_sincos[..., 3:]

    norm = torch.sqrt(sin_vals.pow(2) + cos_vals.pow(2)).clamp(min=1e-6)

    sin_vals = sin_vals / norm

    cos_vals = cos_vals / norm

    if not assume_legacy_omega:

        return torch.cat([sin_vals, cos_vals], dim=-1)

    sin_out = sin_vals.clone()

    cos_out = cos_vals.clone()

    sin_out[..., 2] = -sin_vals[..., 2]

    cos_out[..., 2] = -cos_vals[..., 2]

    return torch.cat([sin_out, cos_out], dim=-1)





def sincos_to_torsion(sincos: torch.Tensor) -> torch.Tensor:

    """Convert (..., 6) sin/cos to (..., 3) angles in degrees."""

    sin_vals = sincos[..., :3]

    cos_vals = sincos[..., 3:]

    angles_rad = torch.atan2(sin_vals, cos_vals)

    return angles_rad * (180.0 / math.pi)





def _nerf_differentiable(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                         length: float, angle_deg: float,
                         dihedral_deg: torch.Tensor) -> torch.Tensor:
    """
    Differentiable NeRF placement where dihedral_deg is a scalar tensor
    that retains its gradient. Bond length and bond angle are constants.
    """
    theta = angle_deg * (math.pi / 180.0)
    chi = dihedral_deg * (math.pi / 180.0)

    bc = c - b
    bc_norm = bc / bc.norm().clamp(min=1e-8)

    ba = a - b
    n = torch.cross(ba, bc, dim=0)
    n_len = n.norm()
    if n_len < 1e-8:
        perp = torch.zeros(3, dtype=c.dtype, device=c.device)
        idx = bc_norm.abs().argmin()
        perp[idx] = 1.0
        n = torch.cross(bc_norm, perp, dim=0)
    n = n / n.norm().clamp(min=1e-8)

    m = torch.cross(n, bc_norm, dim=0)
    m = m / m.norm().clamp(min=1e-8)

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    d_dir = (
        -cos_theta * bc_norm
        + sin_theta * torch.cos(chi) * m
        + sin_theta * torch.sin(chi) * n
    )

    return c + length * d_dir


def build_backbone_from_torsions(phi: torch.Tensor, psi: torch.Tensor,
                                 omega: torch.Tensor,
                                 init_atoms: torch.Tensor = None) -> torch.Tensor:
    """
    Build a protein backbone (N, CA, C) from torsion angles using NeRF.
    This version is fully differentiable w.r.t. phi, psi, omega.

    Args:
        phi: (L,) tensor of phi angles in degrees
        psi: (L,) tensor of psi angles in degrees
        omega: (L,) tensor of omega angles in degrees
        init_atoms: (3, 3) tensor for first residue [N, CA, C]. If None, uses ideal.

    Returns:
        backbone: (L, 3, 3) tensor of [N, CA, C] coordinates
    """
    L = phi.shape[0]
    device = phi.device
    dtype = phi.dtype

    atoms = []

    if init_atoms is None:
        N0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
        CA0 = torch.tensor([BOND_N_CA, 0.0, 0.0], dtype=dtype, device=device)
        n_ca_c_angle_rad = ANGLE_N_CA_C * math.pi / 180.0
        C0 = torch.tensor([
            BOND_N_CA - BOND_CA_C * math.cos(n_ca_c_angle_rad),
            BOND_CA_C * math.sin(n_ca_c_angle_rad),
            0.0
        ], dtype=dtype, device=device)
    else:
        N0 = init_atoms[0]
        CA0 = init_atoms[1]
        C0 = init_atoms[2]

    atoms.append(torch.stack([N0, CA0, C0], dim=0))

    N_i, CA_i, C_i = N0, CA0, C0

    for i in range(L - 1):
        N_next = _nerf_differentiable(N_i, CA_i, C_i, BOND_C_N, ANGLE_CA_C_N, psi[i])
        CA_next = _nerf_differentiable(CA_i, C_i, N_next, BOND_N_CA, ANGLE_C_N_CA, legacy_omega_to_trans180_torch(omega[i]))
        C_next = _nerf_differentiable(C_i, N_next, CA_next, BOND_CA_C, ANGLE_N_CA_C, phi[i + 1])

        atoms.append(torch.stack([N_next, CA_next, C_next], dim=0))
        N_i, CA_i, C_i = N_next, CA_next, C_next

    return torch.stack(atoms, dim=0)


def assign_ss_from_torsions(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:

    """Assign secondary structure labels from phi/psi angles. 0=Helix, 1=Sheet, 2=Coil."""

    L = len(phi)

    ss = np.full(L, 2, dtype=np.int64)



    for i in range(L):

        p, s = phi[i], psi[i]

        if p == 0.0 and s == 0.0:

            continue

        d_helix = (p - HELIX_PHI) ** 2 + (s - HELIX_PSI) ** 2

        d_beta = (p - BETA_PHI) ** 2 + (s - BETA_PSI) ** 2

        if d_helix < 2500:

            ss[i] = 0

        elif d_beta < 2500:

            ss[i] = 1



    return ss

