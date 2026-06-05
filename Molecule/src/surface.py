# -*- coding: utf-8 -*-
"""
Protein surface generation via SDF level-set projection.

Mathematical background (Surface-VQMAE style):
  SDF(x) = -f(x) * log sum_j exp(-||x - a_j|| / sigma_j)
  f(x)   = (sum_j exp(-||x - a_j||) * sigma_j) / (sum_j exp(-||x - a_j||) + eps)

Surface points are obtained by gradient descent to the SDF = r_level isosurface.
Normals are estimated as the normalised gradient of the SDF at surface points.
Inner / trapped points are removed by checking proximity to atom centres.

Self-contained - no imports from other projects.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import to_tensor, normalize, chunk_indices, nchunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDF module
# ---------------------------------------------------------------------------

class SurfaceSDF(nn.Module):
    """
    Differentiable implicit SDF over a protein atomic soup.

    SDF(x) = -f(x) * logsumexp_j( -||x - a_j|| / sigma_j )
    f(x)   = sum_j( exp(-||x-a_j||) * sigma_j ) / ( sum_j exp(-||x-a_j||) + eps )
    """

    def __init__(
        self,
        atom_pos: np.ndarray,
        atom_sigma: np.ndarray,
        device: str = "cpu",
    ):
        super().__init__()
        self.register_buffer("A",   to_tensor(atom_pos,   device))
        self.register_buffer("SIG", to_tensor(atom_sigma, device))

    def forward(self, X: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        """
        X : (Nq, 3)  query points
        returns (Nq,) SDF values
        """
        out = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
        Na = self.A.shape[0]
        for s, e in chunk_indices(X.shape[0], chunk_size):
            xb = X[s:e]                                            # (B, Na)
            d  = torch.cdist(xb, self.A)                          # (B, Na)
            w  = torch.exp(-d)                                     # (B, Na)
            f  = (w * self.SIG.view(1, Na)).sum(dim=1) / (w.sum(dim=1) + 1e-12)
            d_sig = d / (self.SIG.view(1, Na) + 1e-12)
            lse   = torch.logsumexp(-d_sig, dim=1)
            out[s:e] = -f * lse
        return out


# ---------------------------------------------------------------------------
# Level-set projection via gradient descent
# ---------------------------------------------------------------------------

def project_to_levelset(
    X0: torch.Tensor,
    sdf: SurfaceSDF,
    r: float = 1.05,
    iters: int = 200,
    lr: float = 1e-2,
    chunk: int = 2048,
    desc: str = "Projecting",
) -> torch.Tensor:
    """
    Move initial candidate points X0 onto the isosurface SDF(x) = r
    via gradient descent on  L = 0.5 * (SDF(x) - r)^2.

    Parameters
    ----------
    X0    : (N, 3) initial positions
    sdf   : SurfaceSDF module
    r     : target SDF level (Angstroms, default 1.05)
    iters : number of GD steps
    lr    : step size
    chunk : chunk size for gradient computation

    Returns
    -------
    X : (N, 3) projected surface points
    """
    X = X0.clone().detach()
    for _ in range(iters):
        grad = torch.zeros_like(X)
        for s, e in chunk_indices(X.shape[0], chunk):
            xb = X[s:e].clone().detach().requires_grad_(True)
            sdf_b = sdf(xb)
            loss_b = 0.5 * ((sdf_b - r) ** 2).sum()
            g, = torch.autograd.grad(loss_b, xb)
            grad[s:e] = g.detach()
        with torch.no_grad():
            X -= lr * grad
    return X


# ---------------------------------------------------------------------------
# Surface normals via SDF gradient
# ---------------------------------------------------------------------------

def compute_normals(
    X: torch.Tensor,
    sdf: SurfaceSDF,
    chunk: int = 2048,
) -> torch.Tensor:
    """
    Estimate surface normals as the normalised gradient of the SDF.

    Parameters
    ----------
    X   : (N, 3) surface points
    sdf : SurfaceSDF

    Returns
    -------
    N : (N, 3) unit normals
    """
    normals = torch.empty_like(X)
    for s, e in chunk_indices(X.shape[0], chunk):
        xb = X[s:e].clone().detach().requires_grad_(True)
        sdf_b = sdf(xb)
        grads, = torch.autograd.grad(
            outputs=sdf_b,
            inputs=xb,
            grad_outputs=torch.ones_like(sdf_b),
            retain_graph=False,
            create_graph=False,
        )
        normals[s:e] = normalize(grads)
    return normals


# ---------------------------------------------------------------------------
# Remove inner / trapped points
# ---------------------------------------------------------------------------

def remove_inner_points(
    X: torch.Tensor,
    atom_pos: torch.Tensor,
    atom_sigma: torch.Tensor,
    thresh: float = 0.5,
    chunk: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove points that are too close to (buried inside) any atom centre.
    A point x is kept if:  min_j ||x - a_j||  >=  sigma_j - thresh

    Parameters
    ----------
    X          : (N, 3) candidate surface points
    atom_pos   : (Na, 3)
    atom_sigma : (Na,)
    thresh     : tolerance (Angstroms)

    Returns
    -------
    X_clean : (M, 3)   kept surface points
    mask    : (N,) bool tensor  (True = kept)
    """
    keep = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for s, e in chunk_indices(X.shape[0], chunk):
        xb = X[s:e]
        d  = torch.cdist(xb, atom_pos)        # (B, Na)
        min_d, idx = d.min(dim=1)             # (B,)
        sig_near = atom_sigma[idx]            # (B,)
        keep[s:e] = min_d >= (sig_near - thresh)
    return X[keep], keep


# ---------------------------------------------------------------------------
# Full surface generation pipeline
# ---------------------------------------------------------------------------

def generate_surface(
    atom_pos_np: np.ndarray,
    atom_sigma_np: np.ndarray,
    device: str = "cpu",
    eta: int = 20,
    sigma_init: float = 10.0,
    r_level: float = 1.05,
    proj_iters: int = 200,
    proj_lr: float = 1e-2,
    inner_thresh: float = 0.5,
    target_points: int = 50000,
    seed: int = 2024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: atom soup -> surface points + normals.

    Steps
    -----
    1. Sample eta candidate points around each atom (Gaussian noise, sigma=sigma_init).
    2. Project all candidates onto the SDF = r_level isosurface via GD.
    3. Compute normals as normalised SDF gradient.
    4. Remove inner/buried points.
    5. Subsample to at most target_points via FPS.

    Parameters
    ----------
    atom_pos_np   : (Na, 3) float32
    atom_sigma_np : (Na,)   float32
    device        : torch device string
    eta           : number of candidates per atom
    sigma_init    : std of initial Gaussian noise (Angstroms)
    r_level       : SDF isosurface level
    proj_iters    : GD iterations for projection
    proj_lr       : GD step size
    inner_thresh  : cleaning threshold (Angstroms)
    target_points : maximum number of surface points to keep
    seed          : numpy random seed for FPS

    Returns
    -------
    xs : (M, 3) float32  surface point positions
    ns : (M, 3) float32  surface point normals
    """
    from .utils import farthest_point_sampling   # avoid circular at module level

    rng = np.random.default_rng(seed)
    Na  = atom_pos_np.shape[0]

    # Step 1: sample candidates
    centres = np.repeat(atom_pos_np, eta, axis=0)                 # (Na*eta, 3)
    noise   = rng.normal(0.0, sigma_init, centres.shape).astype(np.float32)
    X0_np   = centres + noise

    # Move to device
    X0  = to_tensor(X0_np, device)
    A   = to_tensor(atom_pos_np,   device)
    SIG = to_tensor(atom_sigma_np, device)

    sdf = SurfaceSDF(atom_pos_np, atom_sigma_np, device=device)

    # Step 2: project
    logger.debug("Projecting %d candidates to isosurface (iters=%d)...", X0.shape[0], proj_iters)
    Xs = project_to_levelset(X0, sdf, r=r_level, iters=proj_iters, lr=proj_lr)

    # Step 3: normals
    logger.debug("Computing normals...")
    Ns = compute_normals(Xs, sdf)

    # Step 4: remove inner points
    logger.debug("Cleaning inner points...")
    Xs_clean, keep = remove_inner_points(Xs, A, SIG, thresh=inner_thresh)
    Ns_clean = Ns[keep]

    # Step 5: subsample
    X_np = Xs_clean.detach().cpu().numpy().astype(np.float32)
    N_np = Ns_clean.detach().cpu().numpy().astype(np.float32)

    if X_np.shape[0] > target_points:
        logger.debug("FPS subsampling %d -> %d points...", X_np.shape[0], target_points)
        idx = farthest_point_sampling(X_np, target_points, seed=seed)
        X_np = X_np[idx]
        N_np = N_np[idx]

    return X_np, N_np
