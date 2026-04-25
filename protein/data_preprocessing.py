# -*- coding: utf-8 -*-
"""
Stage-1: Protein Surface Modeling (Surface-VQMAE faithful reproduction)
- Parse PDB -> atoms & residues
- Surface generation via SDF level-set projection (Eq. (1)(2)(3))
- Normals from ∇SDF
- Cleaning (remove trapped/inner points)
- GeoAN neighborhood packaging (store nearest residues for each surface point)
- FPS + KNN to build surface patches
- Morton-order sorting of patch centers
Outputs: one .npz per PDB with all above artifacts.
"""

import os
import sys
import math
import json
import argparse
import warnings
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn

from Bio.PDB import PDBParser, is_aa

# tqdm (progress bar) with graceful fallback
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

ENABLE_TQDM = True  # will be set in main()

def pbar(iterable, total=None, desc=None):
    """Wrapper that returns tqdm iterator when available and enabled."""
    global ENABLE_TQDM, _tqdm
    if ENABLE_TQDM and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, ncols=0, leave=False, dynamic_ncols=True)
    return iterable

# ----------------------------
# Config (defaults reproduce the paper)
# ----------------------------
# PDB_DIR_DEFAULT = "/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/SKEMPI v2.0/PDBs"
PDB_DIR_DEFAULT = "/data/jiangjiaqi/srzhang/InversionDock/Data/Skempi_dataset/Skempiv2"
ELEMENTS = ["C", "H", "O", "N", "S", "Se"]
# approximate vdW radii (Å) used as sigma_a per element; Se fallback to S radius
VDW_SIGMA = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "Se": 1.90,
}
AMINO20 = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]
RES2IDX = {r:i for i,r in enumerate(AMINO20)}

# ----------------------------
# Utility
# ----------------------------

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def chunk_indices(n, chunk):
    for s in range(0, n, chunk):
        e = min(n, s+chunk)
        yield s, e

def nchunks(n, chunk):
    return (n + chunk - 1) // chunk

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def normalize(v: torch.Tensor, eps=1e-8):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)

# Morton 3D code (quantize to 21 bits per axis -> 63-bit Morton)
def _part1by2(n):
    n &= 0x1fffff
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8))  & 0x100f00f00f00f00f
    n = (n | (n << 4))  & 0x10c30c30c30c30c3
    n = (n | (n << 2))  & 0x1249249249249249
    return n

def morton3D(xyz: np.ndarray) -> np.ndarray:
    xyz = np.clip(xyz, 0.0, 1.0)
    q = (xyz * ((1<<21)-1)).astype(np.uint64)
    x, y, z = q[:,0], q[:,1], q[:,2]
    m = _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)
    return m.astype(np.uint64)

# ----------------------------
# Result integrity check (skip if already processed)
# ----------------------------

def is_processed_ok(path: str) -> bool:
    """
    Check if an existing .npz looks complete & sane.
    Required keys: xs, ns, patch_knn_idx, patch_order, meta.
    """
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            required = ["xs", "ns", "patch_knn_idx", "patch_order", "meta"]
            for k in required:
                if k not in data:
                    return False
            xs = data["xs"]; ns = data["ns"]; knn = data["patch_knn_idx"]
            if xs.ndim != 2 or xs.shape[0] == 0: return False
            if ns.shape != xs.shape: return False
            if knn.ndim != 2 or knn.shape[0] == 0: return False
        return True
    except Exception:
        return False

# ----------------------------
# File guards (fix your errors)
# ----------------------------

APPLEDOUBLE_MAGIC = {b"\x00\x05\x16\x07", b"\x00\x05\x16\x00"}  # AppleDouble/AppleSingle

def is_appledouble(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head in APPLEDOUBLE_MAGIC or os.path.basename(path).startswith("._")
    except Exception:
        return False

def looks_like_pdb_text(path: str) -> bool:
    """Cheap sanity check to skip binary/garbled files."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                s = line.lstrip()
                if s.startswith(("ATOM", "HETATM", "MODEL", "HEADER", "SEQRES")):
                    return True
        return False
    except Exception:
        return False

# ----------------------------
# Bio parsing
# ----------------------------

def parse_pdb_atoms_residues(pdb_path: str):
    """
    Returns:
      atom_pos: (Na,3) float32
      atom_sigma: (Na,) float32 per-element vdW radius
      residue_ca_pos: (Nr,3) float32
      residue_type_idx: (Nr,) int64 in [0,19] (others -> -1 filtered)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    atom_pos = []
    atom_sigma = []
    residue_ca_pos = []
    residue_type_idx = []

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                resname = res.get_resname()
                idx = RES2IDX.get(resname, -1)
                if "CA" in res:
                    residue_ca_pos.append(res["CA"].get_coord().astype(np.float32))
                    residue_type_idx.append(idx)
                for atom in res.get_atoms():
                    elem = atom.element.strip().title()
                    if elem == "" or elem not in VDW_SIGMA:
                        continue
                    xyz = atom.get_coord().astype(np.float32)
                    atom_pos.append(xyz)
                    atom_sigma.append(VDW_SIGMA[elem])

    if len(atom_pos) == 0:
        raise ValueError(f"No valid atoms parsed in {pdb_path}")

    atom_pos = np.asarray(atom_pos, dtype=np.float32)
    atom_sigma = np.asarray(atom_sigma, dtype=np.float32)
    if len(residue_ca_pos) == 0:
        residue_ca_pos = atom_pos.copy()
        residue_type_idx = np.full((residue_ca_pos.shape[0],), -1, dtype=np.int64)
    else:
        residue_ca_pos = np.asarray(residue_ca_pos, dtype=np.float32)
        residue_type_idx = np.asarray(residue_type_idx, dtype=np.int64)

    return atom_pos, atom_sigma, residue_ca_pos, residue_type_idx

# ----------------------------
# SDF per Surface-VQMAE (Eq. 1-3)
# ----------------------------

class SurfaceSDF(nn.Module):
    """
    SDF(x) = - f(x) * log sum_j exp(-||x - a_j|| / sigma_j)
    f(x)   = (sum_j exp(-||x - a_j||) * sigma_j) / (sum_j exp(-||x - a_j||))
    """

    def __init__(self, atom_pos: np.ndarray, atom_sigma: np.ndarray, device="cpu"):
        super().__init__()
        self.register_buffer("A", to_tensor(atom_pos, device))         # (Na,3)
        self.register_buffer("SIG", to_tensor(atom_sigma, device))     # (Na,)

    def forward(self, X: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        out = torch.empty((X.shape[0],), dtype=X.dtype, device=X.device)
        Na = self.A.shape[0]
        for s, e in chunk_indices(X.shape[0], chunk_size):
            xb = X[s:e]     # (B,3)
            d = torch.cdist(xb, self.A)                # (B,Na) Euclidean
            w1 = torch.exp(-d)                         # (B,Na)
            num = (w1 * self.SIG.view(1, Na)).sum(dim=1)
            den = w1.sum(dim=1) + 1e-12
            f = num / den
            d_over_sig = d / (self.SIG.view(1, Na) + 1e-12)
            lse = torch.logsumexp(-d_over_sig, dim=1)
            sdf = - f * lse
            out[s:e] = sdf
        return out

def project_to_levelset_gd(
    X0: torch.Tensor, sdf: SurfaceSDF, r=1.05, iters=200, lr=1e-2, chunk=4096, momentum=0.0, desc=None
):
    """
    Manual gradient descent (no torch.optim -> fixes _pytree error on old PyTorch).
    X_{t+1} = X_t - lr * (∂/∂X) [ 0.5 * ||SDF(X) - r||^2 ]
    Optional momentum.
    """
    X = X0.clone().detach()
    V = torch.zeros_like(X)
    for _ in pbar(range(iters), total=iters, desc=desc or "Projecting"):
        grad = torch.zeros_like(X)
        # accumulate gradient chunk by chunk to avoid retain_graph
        for s,e in chunk_indices(X.shape[0], chunk):
            xb = X[s:e].clone().detach().requires_grad_(True)
            sdf_b = sdf(xb)                           # (B,)
            diff = sdf_b - r
            loss_b = 0.5 * (diff * diff).sum()
            g, = torch.autograd.grad(loss_b, xb, retain_graph=False)
            grad[s:e] = g.detach()
        # momentum update
        if momentum > 0:
            V = momentum * V + grad
            step = V
        else:
            step = grad
        with torch.no_grad():
            X -= lr * step
    return X

def sdf_normals(X: torch.Tensor, sdf: SurfaceSDF, chunk=4096, desc=None):
    normals = torch.empty_like(X)
    total = nchunks(X.shape[0], chunk)
    for s, e in pbar(chunk_indices(X.shape[0], chunk), total=total, desc=desc or "Normals"):
        xb = X[s:e].clone().detach().requires_grad_(True)
        sdf_b = sdf(xb)
        grads = torch.autograd.grad(
            outputs=sdf_b,
            inputs=xb,
            grad_outputs=torch.ones_like(sdf_b),
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]
        normals[s:e] = normalize(grads)
    return normals

# ----------------------------
# Cleaning
# ----------------------------

def remove_inner_points(X: torch.Tensor, atom_pos: torch.Tensor, atom_sigma: torch.Tensor, thresh=0.5, chunk=8192, desc=None):
    keep_mask = torch.ones((X.shape[0],), dtype=torch.bool, device=X.device)
    total = nchunks(X.shape[0], chunk)
    for s,e in pbar(chunk_indices(X.shape[0], chunk), total=total, desc=desc or "Cleaning"):
        xb = X[s:e]
        d = torch.cdist(xb, atom_pos)
        min_d, idx = d.min(dim=1)
        sig_near = atom_sigma[idx]
        ok = min_d >= (sig_near - thresh)
        keep_mask[s:e] = ok
    return X[keep_mask], keep_mask

# ----------------------------
# FPS + KNN
# ----------------------------

def farthest_point_sampling(X: np.ndarray, num: int, seed: int = 2023) -> np.ndarray:
    N = X.shape[0]
    if num >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    centers = np.empty((num,), dtype=np.int64)
    centers[0] = rng.integers(0, N)
    dists = np.full((N,), np.inf, dtype=np.float64)
    last = X[centers[0]][None, :]
    for i in range(1, num):
        diff = X - last
        dd = np.sum(diff*diff, axis=1)
        dists = np.minimum(dists, dd)
        centers[i] = int(np.argmax(dists))
        last = X[centers[i]][None, :]
    return centers

def knn_indices(X: np.ndarray, C: np.ndarray, K: int, chunk: int = 4096, desc=None) -> np.ndarray:
    Nc = C.shape[0]
    out = np.empty((Nc, K), dtype=np.int64)
    total = nchunks(Nc, chunk)
    for s,e in pbar(chunk_indices(Nc, chunk), total=total, desc=desc or "KNN"):
        Cb = C[s:e]
        D = np.sqrt(((Cb[:,None,:] - X[None,:,:])**2).sum(axis=-1))
        idx = np.argpartition(D, K-1, axis=1)[:, :K]
        row_sorted = np.take_along_axis(idx, np.argsort(np.take_along_axis(D, idx, axis=1), axis=1), axis=1)
        out[s:e] = row_sorted
    return out

# ----------------------------
# Residue neighborhood for GeoAN packaging
# ----------------------------

def residue_neighbors(surface: np.ndarray, ca_pos: np.ndarray, ca_type_idx: np.ndarray, zeta: int = 16, chunk: int = 4096, desc=None):
    M = surface.shape[0]
    R = ca_pos.shape[0]
    nei_idx = np.empty((M, zeta), dtype=np.int32)
    nei_dist = np.empty((M, zeta), dtype=np.float32)
    nei_type = np.empty((M, zeta), dtype=np.int16)

    total = nchunks(M, chunk)
    for s,e in pbar(chunk_indices(M, chunk), total=total, desc=desc or "Residue NN"):
        Sb = surface[s:e]
        D = np.sqrt(((Sb[:,None,:] - ca_pos[None,:,:])**2).sum(axis=-1))  # (b,R)
        idx = np.argpartition(D, zeta-1, axis=1)[:, :zeta]
        dsel = np.take_along_axis(D, idx, axis=1)
        order = np.argsort(dsel, axis=1)
        idx_sorted = np.take_along_axis(idx, order, axis=1)
        d_sorted = np.take_along_axis(D, idx_sorted, axis=1)
        t_sorted = ca_type_idx[idx_sorted]
        nei_idx[s:e] = idx_sorted.astype(np.int32)
        nei_dist[s:e] = d_sorted.astype(np.float32)
        nei_type[s:e] = t_sorted.astype(np.int16)
    return nei_idx, nei_dist, nei_type

# ----------------------------
# Pipeline
# ----------------------------

def process_one_pdb(
    pdb_path: str,
    out_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eta: int = 20,
    sigma_init: float = 10.0,   # Å
    r_level: float = 1.05,      # Å
    proj_iters: int = 200,
    proj_lr: float = 1e-2,
    inner_thresh: float = 0.5,
    target_points: int = 50000,
    fps_ratio: float = 0.05,
    knn_k: int = 50,
    zeta: int = 16,
    seed: int = 2023,
    overwrite: bool = False,
):
    name = os.path.splitext(os.path.basename(pdb_path))[0]
    out_path = os.path.join(out_dir, f"{name}.npz")
    if (not overwrite) and is_processed_ok(out_path):
        print(f"[SKIP] Exists & valid -> {out_path}")
        return

    print(f"[{name}] Parsing PDB ...")
    atom_pos_np, atom_sig_np, ca_pos_np, ca_type_np = parse_pdb_atoms_residues(pdb_path)

    # torch buffers
    A = to_tensor(atom_pos_np, device)
    SIG = to_tensor(atom_sig_np, device)

    # 1) sampling around atoms
    print(f"[{name}] Sampling η={eta} around {A.shape[0]} atoms ...")
    centers = np.repeat(atom_pos_np, eta, axis=0)
    noise = np.random.normal(loc=0.0, scale=sigma_init, size=centers.shape).astype(np.float32)
    X0_np = centers + noise
    X0 = to_tensor(X0_np, device=device)

    # 2) SDF projection to level-set (manual GD -> no _pytree usage)
    sdf = SurfaceSDF(atom_pos_np, atom_sig_np, device=device)
    Xs = project_to_levelset_gd(
        X0, sdf, r=r_level, iters=proj_iters, lr=proj_lr, momentum=0.0,
        desc=f"[{name}] Projecting"
    )

    # 3) normals
    Ns = sdf_normals(Xs, sdf, desc=f"[{name}] Normals")

    # 4) remove trapped inner points
    Xs_clean, keep = remove_inner_points(Xs, A, SIG, thresh=inner_thresh, desc=f"[{name}] Cleaning")
    Ns_clean = Ns[keep]
    if Xs_clean.shape[0] > target_points:
        print(f"[{name}] Sub-sampling to {target_points} points ...")
        Xcpu = Xs_clean.detach().cpu().numpy()
        idx_keep = farthest_point_sampling(Xcpu, target_points, seed=seed)
        Xs_clean = Xs_clean[idx_keep]
        Ns_clean = Ns_clean[idx_keep]

    X_np = Xs_clean.detach().cpu().numpy().astype(np.float32)
    N_np = Ns_clean.detach().cpu().numpy().astype(np.float32)

    # 5) GeoAN packaging
    nei_idx, nei_dist, nei_type = residue_neighbors(
        X_np, ca_pos_np.astype(np.float32), ca_type_np.astype(np.int16), zeta=zeta, desc=f"[{name}] Residue NN"
    )

    # 6) FPS + KNN patches
    M = X_np.shape[0]
    num_centers = max(1, int(math.ceil(fps_ratio * M)))
    fps_idx = farthest_point_sampling(X_np, num_centers, seed=seed)
    Xc = X_np[fps_idx]
    knn_idx = knn_indices(X_np, Xc, K=knn_k, desc=f"[{name}] KNN")

    # 7) Morton order for patch centers
    mins = Xc.min(axis=0, keepdims=True)
    maxs = Xc.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    Xc_unit = (Xc - mins) / span
    morton = morton3D(Xc_unit)
    order = np.argsort(morton)

    # 8) save
    meta = dict(
        eta=eta, sigma_init=sigma_init, r_level=r_level, proj_iters=proj_iters, proj_lr=proj_lr,
        inner_thresh=inner_thresh, target_points=target_points, fps_ratio=fps_ratio, knn_k=knn_k,
        zeta=zeta, seed=seed, device=device
    )
    print(f"[{name}] Saving -> {out_path}")
    np.savez_compressed(
        out_path,
        xs=X_np,
        ns=N_np,
        ca_pos=ca_pos_np.astype(np.float32),
        ca_type=ca_type_np.astype(np.int16),
        geo_nei_idx=nei_idx,
        geo_nei_dist=nei_dist,
        geo_nei_type=nei_type,
        patch_centers=Xc.astype(np.float32),
        patch_knn_idx=knn_idx.astype(np.int32),
        patch_morton=morton,
        patch_order=order.astype(np.int64),
        fps_idx=fps_idx.astype(np.int64),
        meta=json.dumps(meta),
    )

def main():
    parser = argparse.ArgumentParser(description="Stage-1: Protein Surface Modeling Preprocessing")
    parser.add_argument("--pdb_dir", type=str, default=PDB_DIR_DEFAULT)
    parser.add_argument("--out_dir", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/Processed_Sabdab")
    parser.add_argument("--eta", type=int, default=20)
    parser.add_argument("--sigma_init", type=float, default=10.0)
    parser.add_argument("--r_level", type=float, default=1.05)
    parser.add_argument("--proj_iters", type=int, default=200)
    parser.add_argument("--proj_lr", type=float, default=1e-2)
    parser.add_argument("--inner_thresh", type=float, default=0.5)
    parser.add_argument("--target_points", type=int, default=50000)
    parser.add_argument("--fps_ratio", type=float, default=0.05)
    parser.add_argument("--knn_k", type=int, default=50)
    parser.add_argument("--zeta", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--overwrite", action="store_true", help="Reprocess even if output .npz exists")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars (useful for nohup logs)")
    args = parser.parse_args()

    # enable/disable tqdm globally
    global ENABLE_TQDM
    ENABLE_TQDM = not args.no_progress

    safe_mkdir(args.out_dir)

    # only real .pdb files; skip macOS AppleDouble and non-text PDB
    all_files = sorted(os.listdir(args.pdb_dir))
    pdb_files = []
    for f in all_files:
        if not f.lower().endswith(".pdb"):
            continue
        full = os.path.join(args.pdb_dir, f)
        if is_appledouble(full):
            print(f"[SKIP] AppleDouble resource file: {f}")
            continue
        if not looks_like_pdb_text(full):
            print(f"[SKIP] Not a valid PDB text (no ATOM/HETATM): {f}")
            continue
        pdb_files.append(f)

    if len(pdb_files) == 0:
        print(f"No valid .pdb files found under {args.pdb_dir}")
        sys.exit(1)

    # overall files progress
    for f in pbar(pdb_files, total=len(pdb_files), desc="PDB files"):
        name = os.path.splitext(f)[0]
        out_path = os.path.join(args.out_dir, f"{name}.npz")
        if (not args.overwrite) and is_processed_ok(out_path):
            print(f"[SKIP] Exists & valid -> {out_path}")
            continue
        pdb_path = os.path.join(args.pdb_dir, f)
        try:
            process_one_pdb(
                pdb_path=pdb_path,
                out_dir=args.out_dir,
                device=args.device,
                eta=args.eta,
                sigma_init=args.sigma_init,
                r_level=args.r_level,
                proj_iters=args.proj_iters,
                proj_lr=args.proj_lr,
                inner_thresh=args.inner_thresh,
                target_points=args.target_points,
                fps_ratio=args.fps_ratio,
                knn_k=args.knn_k,
                zeta=args.zeta,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"[WARN] Failed on {f}: {e}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
