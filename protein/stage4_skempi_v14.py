# -*- coding: utf-8 -*-
"""
stage4_skempi_v17_natural_diverse.py  -  分段拓扑 + 长程接触/模板约束 + 全原子骨架优化版本

相比 v13 的核心改进：

[V14-FULLATOM] NeRF 生长直接输出 N/CA/C/O 四原子骨架
  - O 原子由肽键平面几何公式实时推算（C=O 键长 1.229Å，∠CA-C=O 120.8°）
  - 不再有「随机 v2 方向」问题，N/C 由 NeRF 递推保证物理合理性

[V14-SURFACE] 配体骨架动态生成 Surface Patch 特征
  - 直接复用 data_preprocessing.py 的 SDF/投影/FPS/KNN 核心函数
  - 与 Stage3 训练时的特征完全一致（同一套代码路径）
  - 受体特征仍从预处理好的 .npz 读取（不变）

[V14-NOGRAD] 无梯度随机游走骨架优化
  - 放弃 Adam 梯度（SDF 投影不可微分端到端）
  - 改用随机扰动 + Stage3 打分 + 贪婪接受（爬山法）
  - 骨架扰动在 N/CA/C 空间进行，保持肽键几何合理性

[V14-SAVE] 保存全原子 PDB（N/CA/C/O），Rosetta 修复和 MPNN 流程不变

依赖:
  git clone https://github.com/dauparas/ProteinMPNN.git
  权重: ProteinMPNN/vanilla_model_weights/v_48_002.pt
  PyRosetta（可选）
"""
import os
import sys
import math
import re
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage3_skempi_v2 import DockingModel

# ---- ProteinMPNN 路径 ----
def _find_mpnn_root():
    for p in [
        os.environ.get("PROTEINMPNN_PATH", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProteinMPNN"),
    ]:
        if p and os.path.isfile(os.path.join(p, "protein_mpnn_run.py")):
            return os.path.abspath(p)
    return None

MPNN_ROOT = _find_mpnn_root()
if MPNN_ROOT:
    sys.path.insert(0, MPNN_ROOT)
    print(f"[ProteinMPNN] Found: {MPNN_ROOT}")
else:
    print("[WARN] ProteinMPNN not found. Clone: https://github.com/dauparas/ProteinMPNN")

try:
    import pyrosetta
    from pyrosetta import rosetta
    PYROSETTA_AVAILABLE = True
except ImportError:
    PYROSETTA_AVAILABLE = False
    print("[WARN] PyRosetta not found; backbone fix skipped.")


# ============================================================
# Section 0: 从 data_preprocessing.py 复用的核心函数
# （完整移植，保证与 Stage3 训练时特征完全一致）
# ============================================================

VDW_SIGMA = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "Se": 1.90}


def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _chunk_indices(n, chunk):
    for s in range(0, n, chunk):
        yield s, min(n, s + chunk)


def _normalize(v, eps=1e-8):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


class _SurfaceSDF(nn.Module):
    """与 data_preprocessing.SurfaceSDF 完全相同。"""

    def __init__(self, atom_pos: np.ndarray, atom_sigma: np.ndarray, device):
        super().__init__()
        self.register_buffer("A", _to_tensor(atom_pos, device))
        self.register_buffer("SIG", _to_tensor(atom_sigma, device))

    def forward(self, X: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        out = torch.empty((X.shape[0],), dtype=X.dtype, device=X.device)
        Na = self.A.shape[0]
        for s, e in _chunk_indices(X.shape[0], chunk_size):
            xb = X[s:e]
            d = torch.cdist(xb, self.A)
            w1 = torch.exp(-d)
            num = (w1 * self.SIG.view(1, Na)).sum(dim=1)
            den = w1.sum(dim=1) + 1e-12
            f = num / den
            d_over_sig = d / (self.SIG.view(1, Na) + 1e-12)
            lse = torch.logsumexp(-d_over_sig, dim=1)
            out[s:e] = -f * lse
        return out


def _project_to_levelset(X0: torch.Tensor, sdf: _SurfaceSDF,
                          r=1.05, iters=60, lr=1e-2, chunk=4096) -> torch.Tensor:
    """SDF 等值面投影，iters 默认 60（相比原版 200 加速，精度略降但够用）。"""
    X = X0.clone().detach()
    for _ in range(iters):
        grad = torch.zeros_like(X)
        for s, e in _chunk_indices(X.shape[0], chunk):
            xb = X[s:e].clone().detach().requires_grad_(True)
            sdf_b = sdf(xb)
            loss_b = 0.5 * ((sdf_b - r) ** 2).sum()
            g, = torch.autograd.grad(loss_b, xb)
            grad[s:e] = g.detach()
        with torch.no_grad():
            X -= lr * grad
    return X


def _sdf_normals(X: torch.Tensor, sdf: _SurfaceSDF, chunk=4096) -> torch.Tensor:
    normals = torch.empty_like(X)
    for s, e in _chunk_indices(X.shape[0], chunk):
        xb = X[s:e].clone().detach().requires_grad_(True)
        sdf_b = sdf(xb)
        grads = torch.autograd.grad(
            outputs=sdf_b, inputs=xb,
            grad_outputs=torch.ones_like(sdf_b),
            retain_graph=False, create_graph=False,
        )[0]
        normals[s:e] = _normalize(grads)
    return normals


def _remove_inner_points(X: torch.Tensor, A: torch.Tensor, SIG: torch.Tensor,
                          thresh=0.5, chunk=8192):
    keep = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for s, e in _chunk_indices(X.shape[0], chunk):
        xb = X[s:e]
        d = torch.cdist(xb, A)
        min_d, idx = d.min(dim=1)
        ok = min_d >= (SIG[idx] - thresh)
        keep[s:e] = ok
    return X[keep], keep


def _fps(X: np.ndarray, num: int, seed: int = 0) -> np.ndarray:
    N = X.shape[0]
    if num >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    centers = np.empty((num,), dtype=np.int64)
    centers[0] = rng.integers(0, N)
    dists = np.full((N,), np.inf, dtype=np.float64)
    last = X[centers[0]][None, :]
    for i in range(1, num):
        dd = np.sum((X - last) ** 2, axis=1)
        dists = np.minimum(dists, dd)
        centers[i] = int(np.argmax(dists))
        last = X[centers[i]][None, :]
    return centers


def _knn(X: np.ndarray, C: np.ndarray, K: int, chunk: int = 4096) -> np.ndarray:
    Nc = C.shape[0]
    out = np.empty((Nc, K), dtype=np.int64)
    for s, e in _chunk_indices(Nc, chunk):
        Cb = C[s:e]
        D = np.sqrt(((Cb[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        idx = np.argpartition(D, min(K, D.shape[1]) - 1, axis=1)[:, :K]
        row_sorted = np.take_along_axis(
            idx, np.argsort(np.take_along_axis(D, idx, axis=1), axis=1), axis=1
        )
        out[s:e] = row_sorted
    return out


def build_surface_patch_features(
    atom_pos_np: np.ndarray,
    atom_sigma_np: np.ndarray,
    device: torch.device,
    eta: int = 8,
    r_level: float = 1.05,
    proj_iters: int = 60,
    proj_lr: float = 1e-2,
    inner_thresh: float = 0.5,
    fps_ratio: float = 0.05,
    knn_k: int = 50,
    max_patches: int = 512,
    seed: int = 0,
) -> tuple:
    """
    给定全原子坐标，动态生成与 Stage3 训练完全一致的 surface patch 特征。

    返回:
        feats   : Tensor [1, N_patch, K, 6]  (rel_xyz || normals)
        centers : Tensor [1, N_patch, 3]
        xs_t    : Tensor [M, 3]  表面点云（用于 clash 检查）
    """
    A = _to_tensor(atom_pos_np, device)
    SIG = _to_tensor(atom_sigma_np, device)
    sdf = _SurfaceSDF(atom_pos_np, atom_sigma_np, device)

    # 1. 在原子周围随机采样初始点
    centers_np = np.repeat(atom_pos_np, eta, axis=0)
    noise = np.random.randn(*centers_np.shape).astype(np.float32) * 5.0
    X0 = _to_tensor(centers_np + noise, device)

    # 2. 投影到等值面
    Xs = _project_to_levelset(X0, sdf, r=r_level, iters=proj_iters, lr=proj_lr)

    # 3. 计算法向量
    Ns = _sdf_normals(Xs, sdf)

    # 4. 去除内部点
    Xs, keep = _remove_inner_points(Xs, A, SIG, thresh=inner_thresh)
    Ns = Ns[keep]

    X_np = Xs.detach().cpu().numpy().astype(np.float32)
    N_np = Ns.detach().cpu().numpy().astype(np.float32)

    # 5. FPS 选 patch 中心
    M = X_np.shape[0]
    num_c = max(1, min(max_patches, int(math.ceil(fps_ratio * M))))
    fps_idx = _fps(X_np, num_c, seed=seed)
    Xc = X_np[fps_idx]             # [num_c, 3]

    # 6. KNN
    knn_idx = _knn(X_np, Xc, K=knn_k)   # [num_c, K]

    # 7. 构建 feats（与 Stage3 DataLoader 完全相同）
    rel_xyz = X_np[knn_idx] - Xc[:, None, :]   # [num_c, K, 3]
    nrm     = N_np[knn_idx]                     # [num_c, K, 3]
    feat_np = np.concatenate([rel_xyz, nrm], axis=-1).astype(np.float32)  # [num_c, K, 6]

    feats   = torch.from_numpy(feat_np).unsqueeze(0).to(device)    # [1, num_c, K, 6]
    centers = torch.from_numpy(Xc).unsqueeze(0).to(device)         # [1, num_c, 3]
    xs_t    = torch.from_numpy(X_np).to(device)                    # [M, 3]
    return feats, centers, xs_t


# ============================================================
# Section 1: NeRF 全原子骨架生长（N/CA/C/O）
# ============================================================

def _tensor_all_finite(x):
    try:
        return bool(torch.isfinite(x).all().item())
    except Exception:
        return False


def _place_atom_nerf(a, b, c, bond_length,
                     bond_angle_deg, dihedral_deg):
    eps = 1e-8
    device, dtype = c.device, c.dtype
    theta = torch.deg2rad(torch.tensor(bond_angle_deg, device=device, dtype=dtype))
    chi   = torch.deg2rad(torch.tensor(dihedral_deg,   device=device, dtype=dtype))
    bc_u  = (c - b) / (torch.norm(c - b) + eps)
    ba    = a - b
    n     = ba - torch.dot(ba, bc_u) * bc_u
    nn    = torch.norm(n)
    if float(nn) < 1e-6:
        perp = torch.randn(3, device=device, dtype=dtype)
        n = perp - torch.dot(perp, bc_u) * bc_u
        nn = torch.norm(n) + eps
    n_u = n / (nn + eps)
    m_u = torch.linalg.cross(bc_u, n_u)
    d = bond_length * (-torch.cos(theta) * bc_u +
                       torch.sin(theta) * (torch.cos(chi) * n_u + torch.sin(chi) * m_u))
    return c + d


def _place_o_from_ncac(N, CA, C):
    """
    根据 N-CA-C 三原子推算 O 坐标。
    C=O 键长 1.229Å，∠CA-C=O = 120.8°，O 在肽键平面内（与 N 同侧的反方向）。
    """
    eps = 1e-8
    b = CA - N
    c = C - CA
    bc = c / (torch.norm(c) + eps)
    n_vec = torch.linalg.cross(b, c)
    n_norm = torch.norm(n_vec)
    if float(n_norm) < 1e-6:
        return C + bc * 1.229
    n_vec = n_vec / n_norm
    angle = math.radians(120.8)
    perp = torch.linalg.cross(n_vec, bc)
    perp = perp / (torch.norm(perp) + eps)
    o_vec = bc * math.cos(math.pi - angle) + perp * math.sin(math.pi - angle)
    return C + o_vec * 1.229


def _build_first_residue_fullatom(pocket_center, device, dtype=torch.float32):
    """
    在 pocket_center 附近放置第一个残基的 N/CA/C/O 四原子。
    返回 [4, 3] tensor，顺序：N, CA, C, O
    """
    eps = 1e-8
    ca = pocket_center.to(device=device, dtype=dtype)
    for _ in range(10):
        v1 = torch.randn(3, device=device, dtype=dtype)
        v1 = v1 / (torch.norm(v1) + eps)
        v2 = torch.randn(3, device=device, dtype=dtype)
        v2 = v2 - torch.dot(v2, v1) * v1
        if float(torch.norm(v2)) > 1e-4:
            v2 = v2 / (torch.norm(v2) + eps)
            break
    ang = torch.deg2rad(torch.tensor(111.2, device=device, dtype=dtype))
    N  = ca - 1.458 * v1
    CA = ca
    C  = ca + 1.525 * (torch.cos(ang) * v1 + torch.sin(ang) * v2)
    O  = _place_o_from_ncac(N, CA, C)
    return torch.stack([N, CA, C, O], dim=0)  # [4, 3]


def _build_next_residue_fullatom(prev_N, prev_CA, prev_C, prev_O,
                                  phi_deg=-65.0, psi_deg=-40.0, omega_deg=180.0):
    """
    由上一个残基的 N/CA/C/O 生长下一个残基的 N/CA/C/O。
    返回 [4, 3] tensor，顺序：N, CA, C, O
    """
    N_i  = _place_atom_nerf(prev_N,  prev_CA, prev_C, 1.329, 116.2, psi_deg)
    CA_i = _place_atom_nerf(prev_CA, prev_C,  N_i,   1.458, 121.7, omega_deg)
    C_i  = _place_atom_nerf(prev_C,  N_i,    CA_i,  1.525, 111.2, phi_deg)
    O_i  = _place_o_from_ncac(N_i, CA_i, C_i)
    return torch.stack([N_i, CA_i, C_i, O_i], dim=0)  # [4, 3]


def _build_default_segment_spec(seq_len, topology_mode="auto"):
    """
    生成默认拓扑模板。
    H=helix, L=loop, E=strand（当前优化仍主要偏向 helix/loop，strand 作为可选预留）
    改进点：默认模板更加偏向“可回折”的三级折叠，而不是单纯长螺旋。
    """
    topology_mode = (topology_mode or "auto").lower()
    if topology_mode == "single_helix":
        return [("H", int(seq_len))]
    if topology_mode == "helix_loop_helix":
        h1 = max(8, int(round(seq_len * 0.40)))
        l1 = max(3, int(round(seq_len * 0.12)))
        h2 = max(8, seq_len - h1 - l1)
        return [("H", h1), ("L", l1), ("H", h2)]
    if topology_mode in ("three_helix", "helix_loop_helix_loop_helix", "auto"):
        if seq_len < 28:
            h1 = max(8, int(round(seq_len * 0.40)))
            l1 = max(3, int(round(seq_len * 0.12)))
            h2 = max(8, seq_len - h1 - l1)
            return [("H", h1), ("L", l1), ("H", h2)]
        l1 = max(3, int(round(seq_len * 0.10)))
        l2 = max(3, int(round(seq_len * 0.10)))
        remain = seq_len - l1 - l2
        h1 = max(8, int(round(remain * 0.34)))
        h2 = max(8, int(round(remain * 0.32)))
        h3 = max(8, remain - h1 - h2)
        total = h1 + l1 + h2 + l2 + h3
        h3 += seq_len - total
        return [("H", h1), ("L", l1), ("H", h2), ("L", l2), ("H", h3)]
    raise ValueError(f"Unknown topology_mode: {topology_mode}")


def _parse_segment_spec(segment_spec, seq_len):
    """
    解析字符串模板，如 H10-L4-H11-L4-H10。
    """
    if not segment_spec:
        return None
    tokens = re.findall(r"[HLEhle]\d+", segment_spec)
    segs = []
    for p in tokens:
        m = re.match(r"^([HLEhle])(\d+)$", p)
        if not m:
            raise ValueError(f"Bad segment spec token: {p}. Use format like H10-L4-H10")
        segs.append((m.group(1).upper(), int(m.group(2))))
    total = sum(x[1] for x in segs)
    if total != int(seq_len):
        raise ValueError(f"segment_spec total length {total} != seq_len {seq_len}")
    return segs


def _expand_segment_schedule(seq_len, topology_mode="auto", segment_spec=None,
                             helix_phi=-62.0, helix_psi=-43.0,
                             loop_phi=-82.0, loop_psi=145.0,
                             strand_phi=-130.0, strand_psi=135.0):
    """
    返回:
        phi_ref, psi_ref, omega_ref: shape [seq_len-1]
        torsion_std_scale: 每个位点扰动系数
        seg_meta: 每个残基所属 segment 信息
        segments: [(stype, start, end_exclusive)]
    注意 torsion 数量是 n_res-1，因为每一步长一个新残基。
    """
    segs = _parse_segment_spec(segment_spec, seq_len)
    if segs is None:
        segs = _build_default_segment_spec(seq_len, topology_mode)

    residue_types = []
    segments = []
    cursor = 0
    for stype, seg_len in segs:
        start = cursor
        end = cursor + int(seg_len)
        segments.append((stype, start, end))
        residue_types.extend([stype] * int(seg_len))
        cursor = end
    if len(residue_types) != seq_len:
        raise ValueError(f"Expanded segment schedule length {len(residue_types)} != seq_len {seq_len}")

    n_tors = max(0, seq_len - 1)
    phi_ref = np.zeros((n_tors,), dtype=np.float32)
    psi_ref = np.zeros((n_tors,), dtype=np.float32)
    omega_ref = np.full((n_tors,), 180.0, dtype=np.float32)
    torsion_std_scale = np.ones((n_tors,), dtype=np.float32)

    for i in range(n_tors):
        stype = residue_types[i + 1]
        if stype == "H":
            phi_ref[i] = helix_phi
            psi_ref[i] = helix_psi
            torsion_std_scale[i] = 0.75
        elif stype == "L":
            phi_ref[i] = loop_phi
            psi_ref[i] = loop_psi
            torsion_std_scale[i] = 1.60
        elif stype == "E":
            phi_ref[i] = strand_phi
            psi_ref[i] = strand_psi
            torsion_std_scale[i] = 1.15
        else:
            phi_ref[i] = helix_phi
            psi_ref[i] = helix_psi
            torsion_std_scale[i] = 1.0

    seg_meta = {
        "residue_types": residue_types,
        "segments": segments,
        "segment_spec": "-".join([f"{t}{e-s}" for t, s, e in segments]),
    }
    seg_meta["topology_blueprint"] = _build_topology_blueprint(seg_meta)
    return phi_ref, psi_ref, omega_ref, torsion_std_scale, seg_meta


def _segment_anchor_positions(s, e, n_points=4, margin=1):
    L = int(e - s)
    if L <= 2 * margin:
        return []
    left = s + margin
    right = e - 1 - margin
    if right < left:
        return []
    if n_points <= 1 or right == left:
        return [int((left + right) // 2)]
    vals = np.linspace(left, right, num=n_points)
    out = []
    seen = set()
    for x in vals:
        xi = int(round(float(x)))
        xi = min(max(xi, left), right)
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
    return out


def _build_topology_blueprint(seg_meta):
    """
    根据 segment 拓扑构造三级结构 blueprint：
      - target contact graph
      - core / surface residue 先验
    """
    if seg_meta is None:
        return {"contact_pairs": [], "core_residues": [], "surface_residues": [], "pair_groups": []}

    segments = seg_meta.get("segments", [])
    helix_segments = [(idx, s, e) for idx, (t, s, e) in enumerate(segments) if t == "H" and (e - s) >= 6]
    pair_groups = []
    contact_pairs = []

    def add_pairs(seg_a, seg_b, d0=9.2, n_points=4, mode="anti"):
        idx_a, sa, ea = seg_a
        idx_b, sb, eb = seg_b
        aa = _segment_anchor_positions(sa, ea, n_points=n_points, margin=1)
        bb = _segment_anchor_positions(sb, eb, n_points=n_points, margin=1)
        if not aa or not bb:
            return
        if mode == "anti":
            bb2 = list(reversed(bb))
        else:
            bb2 = bb
        local_pairs = []
        for i, j in zip(aa, bb2):
            if abs(i - j) >= 4:
                local_pairs.append((i, j, d0))
        if local_pairs:
            pair_groups.append({
                "segments": (idx_a, idx_b),
                "pairs": local_pairs,
                "mode": mode,
                "target_dist": d0,
            })
            contact_pairs.extend(local_pairs)

    if len(helix_segments) >= 2:
        add_pairs(helix_segments[0], helix_segments[1], d0=9.0, n_points=5, mode="anti")
    if len(helix_segments) >= 3:
        add_pairs(helix_segments[1], helix_segments[2], d0=9.0, n_points=5, mode="anti")
        add_pairs(helix_segments[0], helix_segments[2], d0=10.5, n_points=3, mode="parallel")

    core_res = []
    surface_res = []
    for seg_idx, (t, s, e) in enumerate(segments):
        L = e - s
        if L <= 0:
            continue
        mid = int((s + e - 1) // 2)
        if t == "H":
            core_res.extend(_segment_anchor_positions(s, e, n_points=3 if L >= 9 else 2, margin=2 if L >= 8 else 1))
            surface_res.extend([s, min(s + 1, e - 1), max(e - 2, s), e - 1])
            if seg_idx == 1:
                core_res.append(mid)
        elif t == "L":
            surface_res.extend(list(range(s, e)))
        elif t == "E":
            core_res.append(mid)
            surface_res.extend([s, e - 1])

    dedup_core = []
    seen = set()
    for x in core_res:
        if x not in seen:
            dedup_core.append(x)
            seen.add(x)
    dedup_surface = []
    seen = set(dedup_core)
    for x in surface_res:
        if x not in seen:
            dedup_surface.append(x)
            seen.add(x)

    return {
        "contact_pairs": contact_pairs,
        "core_residues": dedup_core,
        "surface_residues": dedup_surface,
        "pair_groups": pair_groups,
    }


def nerf_grow_fullatom(seq_len, pocket_center, device,
                        phi_noise=15.0, psi_noise=15.0,
                        min_ca_dist=2.0,
                        topology_mode="auto",
                        segment_spec=None):
    """
    从 pocket_center 出发，NeRF 生长全原子骨架。
    相比原版，支持分段拓扑模板（如 H-L-H / H-L-H-L-H）。
    返回:
        coords: Tensor [n_res, 4, 3]
        schedule: dict，包含 phi/psi/omega 参考值与拓扑信息
    """
    first = _build_first_residue_fullatom(pocket_center, device)
    residues = [first]

    phi_ref, psi_ref, omega_ref, torsion_std_scale, seg_meta = _expand_segment_schedule(
        seq_len, topology_mode=topology_mode, segment_spec=segment_spec
    )
    n_tors = max(0, seq_len - 1)

    for i in range(n_tors):
        prev = residues[-1]
        scale = float(torsion_std_scale[i])
        phi = float(phi_ref[i] + np.random.randn() * phi_noise * scale)
        psi = float(psi_ref[i] + np.random.randn() * psi_noise * scale)
        omega = float(omega_ref[i] + np.random.randn() * (3.0 if seg_meta["residue_types"][i + 1] == "H" else 6.0))
        new_res = _build_next_residue_fullatom(
            prev[0], prev[1], prev[2], prev[3],
            phi_deg=phi, psi_deg=psi, omega_deg=omega,
        )
        if not _tensor_all_finite(new_res):
            break
        new_ca = new_res[1]
        prev_cas = torch.stack([r[1] for r in residues], dim=0)
        if float(torch.norm(prev_cas - new_ca, dim=1).min()) < min_ca_dist:
            break
        residues.append(new_res)

    coords = torch.stack(residues, dim=0)
    actual_len = coords.shape[0]
    if actual_len != seq_len:
        actual_tors = max(0, actual_len - 1)
        phi_ref = phi_ref[:actual_tors]
        psi_ref = psi_ref[:actual_tors]
        omega_ref = omega_ref[:actual_tors]
        torsion_std_scale = torsion_std_scale[:actual_tors]
        seg_meta["residue_types"] = seg_meta["residue_types"][:actual_len]
        new_segments = []
        for stype, s, e in seg_meta["segments"]:
            if s >= actual_len:
                break
            new_segments.append((stype, s, min(e, actual_len)))
        seg_meta["segments"] = new_segments
        seg_meta["segment_spec"] = "-".join([f"{t}{e-s}" for t, s, e in new_segments])

    schedule = {
        "phi_ref": phi_ref,
        "psi_ref": psi_ref,
        "omega_ref": omega_ref,
        "torsion_std_scale": torsion_std_scale,
        "seg_meta": seg_meta,
        "topology_mode": topology_mode,
        "segment_spec": seg_meta["segment_spec"],
    }
    return coords, schedule


# ============================================================
# Section 2: 从全原子骨架提取原子信息（用于 SDF）
# ============================================================

def _backbone_to_atoms(coords_ncaco):
    """
    coords_ncaco: Tensor [n_res, 4, 3]（N/CA/C/O）
    返回:
        atom_pos_np  : [n_atom, 3] float32
        atom_sigma_np: [n_atom]  float32  (N=1.55, CA/C=1.70, O=1.52)
    """
    SIGMA_MAP = {0: 1.55, 1: 1.70, 2: 1.70, 3: 1.52}  # N, CA, C, O
    n_res = coords_ncaco.shape[0]
    pos_list = []
    sig_list = []
    coords_np = coords_ncaco.detach().cpu().numpy()
    for i in range(n_res):
        for j in range(4):  # N, CA, C, O
            pos_list.append(coords_np[i, j])
            sig_list.append(SIGMA_MAP[j])
    return (np.array(pos_list, dtype=np.float32),
            np.array(sig_list, dtype=np.float32))


# ============================================================
# Section 3: Stage3 无梯度骨架优化（分段拓扑 + 长程接触/模板约束）
# ============================================================

def _stage3_score_surface(coords_ncaco, model, rec_feats, rec_centers, rec_mask,
                           K=50, fps_ratio=0.05, eta=8, proj_iters=60,
                           device=None):
    """
    给定全原子骨架，动态生成 surface patch 特征，送进 Stage3 打分。
    返回 bind_logit (float)。
    注意：build_surface_patch_features 内部用 autograd 计算 SDF 法向量，
    必须在 torch.enable_grad() 上下文中运行；Stage3 前向推断时再禁用梯度。
    """
    if device is None:
        device = coords_ncaco.device
    atom_pos, atom_sigma = _backbone_to_atoms(coords_ncaco)
    # SDF 投影和法向量计算需要梯度，必须在 enable_grad 上下文中
    try:
        with torch.enable_grad():
            feats, centers, _ = build_surface_patch_features(
                atom_pos, atom_sigma, device,
                eta=eta, proj_iters=proj_iters, fps_ratio=fps_ratio, knn_k=K,
            )
    except Exception as e:
        print(f"    [WARN] surface build failed: {e}")
        return -999.0
    # Stage3 前向推断不需要梯度
    n_patch = feats.shape[1]
    lig_mask = torch.zeros(1, n_patch, dtype=torch.bool, device=device)
    try:
        with torch.no_grad():
            bl = model(rec_feats, rec_centers, rec_mask,
                       feats, centers, lig_mask)["bind_logit"]
        return float(bl.item())
    except Exception as e:
        print(f"    [WARN] Stage3 score failed: {e}")
        return -999.0



def _safe_unit(v, eps=1e-8):
    return v / (torch.norm(v) + eps)


def _orthogonal_unit(v, ref=None, eps=1e-8):
    if ref is None:
        ref = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype)
        if abs(float(torch.dot(_safe_unit(v), ref))) > 0.9:
            ref = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype)
    u = ref - torch.dot(ref, v) * v
    nu = torch.norm(u)
    if float(nu) < 1e-6:
        alt = torch.tensor([0.0, 0.0, 1.0], device=v.device, dtype=v.dtype)
        u = alt - torch.dot(alt, v) * v
        nu = torch.norm(u) + eps
    return u / (nu + eps)


def _seed_frame_from_coords(coords):
    """
    从初始全原子骨架提取首残基局部参考系，避免重建时引入随机 up 向量。
    返回:
        first_ca: [3]
        forward : [3]
        up      : [3]
    """
    first = coords[0]
    first_ca = first[1].clone()
    if coords.shape[0] > 1:
        forward = _safe_unit(coords[1, 1] - coords[0, 1])
    else:
        forward = _safe_unit(first[2] - first[1])

    plane_up = first[2] - first[1]
    plane_up = plane_up - torch.dot(plane_up, forward) * forward
    if float(torch.norm(plane_up)) < 1e-6:
        plane_up = first[0] - first[1]
        plane_up = plane_up - torch.dot(plane_up, forward) * forward
    up = _orthogonal_unit(forward, ref=plane_up)
    return first_ca, forward, up


def _build_first_residue_from_frame(ca, forward, up):
    ang = torch.deg2rad(torch.tensor(111.2, device=ca.device, dtype=ca.dtype))
    N = ca - 1.458 * forward
    C = ca + 1.525 * (torch.cos(ang) * forward + torch.sin(ang) * up)
    O = _place_o_from_ncac(N, ca, C)
    return torch.stack([N, ca, C, O], dim=0)


def build_chain_from_torsions(first_ca, forward, up, phi_deg, psi_deg, omega_deg):
    """
    通过首残基参考系 + torsion 序列，连续 NeRF 重建整条链。
    这样优化变量始终是 phi/psi/omega，而不是把 CA 拉散后再随机补原子。
    """
    first = _build_first_residue_from_frame(first_ca, _safe_unit(forward), _safe_unit(up))
    residues = [first]
    for i in range(1, len(phi_deg) + 1):
        prev = residues[-1]
        new_res = _build_next_residue_fullatom(
            prev[0], prev[1], prev[2], prev[3],
            phi_deg=float(phi_deg[i - 1]),
            psi_deg=float(psi_deg[i - 1]),
            omega_deg=float(omega_deg[i - 1]),
        )
        residues.append(new_res)
    return torch.stack(residues, dim=0)


def _pocket_penalty(coords, pocket_center):
    ca_coords = coords[:, 1, :]
    centroid = ca_coords.mean(dim=0)
    return float(F.mse_loss(centroid, pocket_center.to(coords.device)).item())


def _ca_step_penalty(coords, target=3.8):
    ca = coords[:, 1, :]
    if ca.shape[0] < 2:
        return 0.0
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    return float(((d - target) ** 2).mean().item())


def _ca_clash_penalty(coords, min_allowed=3.0, exclude_neighbours=2):
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if n < 4:
        return 0.0
    dmat = torch.cdist(ca, ca)
    idx = torch.arange(n, device=ca.device)
    seq_sep = torch.abs(idx[:, None] - idx[None, :])
    mask = seq_sep > exclude_neighbours
    viol = F.relu(min_allowed - dmat[mask])
    if viol.numel() == 0:
        return 0.0
    return float((viol ** 2).mean().item())


def _compactness_penalty(coords):
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if n < 3:
        return 0.0
    center = ca.mean(dim=0, keepdim=True)
    rg = torch.sqrt(((ca - center) ** 2).sum(dim=-1).mean() + 1e-8)
    # 对短肽给一个较宽松的紧致度上界，避免完全拉成大线圈
    rg_target = 2.0 + 0.22 * math.sqrt(float(n))
    return float(F.relu(rg - rg_target).pow(2).item())


def _torsion_prior_penalty(phi_deg, psi_deg, omega_deg,
                           phi_ref=-65.0, psi_ref=-40.0, omega_ref=180.0):
    phi_deg = np.asarray(phi_deg, dtype=np.float32)
    psi_deg = np.asarray(psi_deg, dtype=np.float32)
    omega_deg = np.asarray(omega_deg, dtype=np.float32)
    phi_ref = np.asarray(phi_ref, dtype=np.float32) if np.ndim(phi_ref) > 0 else float(phi_ref)
    psi_ref = np.asarray(psi_ref, dtype=np.float32) if np.ndim(psi_ref) > 0 else float(psi_ref)
    omega_ref = np.asarray(omega_ref, dtype=np.float32) if np.ndim(omega_ref) > 0 else float(omega_ref)
    pen = ((phi_deg - phi_ref) ** 2).mean()
    pen += ((psi_deg - psi_ref) ** 2).mean()
    pen += 0.25 * ((omega_deg - omega_ref) ** 2).mean()
    return float(pen)


def _helix_reward(coords):
    """
    用 CA(i,i+3)/(i,i+4) 的距离模式近似鼓励 alpha-helix。
    这不是严格 DSSP，但在无序 coil 与短螺旋之间足够提供偏置。
    """
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if n < 5:
        return 0.0
    score = 0.0
    count = 0
    if n >= 4:
        d3 = torch.norm(ca[3:] - ca[:-3], dim=-1)
        score += float(torch.exp(-((d3 - 5.1) ** 2) / 1.5).mean().item())
        count += 1
    if n >= 5:
        d4 = torch.norm(ca[4:] - ca[:-4], dim=-1)
        score += float(torch.exp(-((d4 - 6.2) ** 2) / 1.8).mean().item())
        count += 1
    return score / max(count, 1)


def _interval_contact_score(dist, good_lo=7.0, good_hi=11.0, far_cut=13.0, clash_lo=6.0):
    if dist < clash_lo:
        return -((clash_lo - dist) / clash_lo) ** 2
    if dist <= good_lo:
        return 0.4 + 0.6 * (dist - clash_lo) / max(good_lo - clash_lo, 1e-6)
    if dist <= good_hi:
        return 1.0
    if dist <= far_cut:
        return 1.0 - (dist - good_hi) / max(far_cut - good_hi, 1e-6)
    return 0.0


def _long_range_contact_reward(coords, min_seq_sep=6, blueprint=None):
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if n <= min_seq_sep + 1:
        return 0.0
    pairs = []
    if blueprint is not None:
        for i, j, _ in blueprint.get("contact_pairs", []):
            if 0 <= i < n and 0 <= j < n and abs(i - j) >= min_seq_sep:
                pairs.append((i, j))
    if not pairs:
        return 0.0
    scores = []
    for i, j in pairs:
        d = float(torch.norm(ca[i] - ca[j]).item())
        scores.append(_interval_contact_score(d, good_lo=7.5, good_hi=10.8, far_cut=13.0, clash_lo=6.0))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _end_distance_penalty(coords, target_end_dist=10.0):
    ca = coords[:, 1, :]
    if ca.shape[0] < 2:
        return 0.0
    d = torch.norm(ca[0] - ca[-1])
    return float(((d - target_end_dist) ** 2).item())


def _build_template_contact_pairs(seg_meta):
    """
    优先使用 topology blueprint 中更密、更明确的稀疏接触图。
    若 blueprint 不存在，则回退到旧式自动模板。
    """
    if seg_meta is None:
        return []
    blueprint = seg_meta.get("topology_blueprint")
    if blueprint and blueprint.get("contact_pairs"):
        return list(blueprint["contact_pairs"])

    segments = seg_meta.get("segments", [])
    helix_segments = [(s, e) for t, s, e in segments if t == "H" and (e - s) >= 6]
    pairs = []
    if len(helix_segments) >= 2:
        a1 = _segment_anchor_positions(*helix_segments[0], n_points=4, margin=1)
        a2 = _segment_anchor_positions(*helix_segments[1], n_points=4, margin=1)
        for i, j in zip(a1, reversed(a2)):
            if abs(i - j) >= 4:
                pairs.append((i, j, 9.0))
    if len(helix_segments) >= 3:
        a2 = _segment_anchor_positions(*helix_segments[1], n_points=4, margin=1)
        a3 = _segment_anchor_positions(*helix_segments[2], n_points=4, margin=1)
        a1 = _segment_anchor_positions(*helix_segments[0], n_points=3, margin=1)
        for i, j in zip(a2, reversed(a3)):
            if abs(i - j) >= 4:
                pairs.append((i, j, 9.0))
        for i, j in zip(a1, a3):
            if abs(i - j) >= 4:
                pairs.append((i, j, 10.5))
    return pairs



def _default_target_end_dist(seg_meta):
    if seg_meta is None:
        return None
    segments = seg_meta.get("segments", [])
    helix_n = sum(1 for t, s, e in segments if t == "H")
    if helix_n >= 3:
        return 10.0
    if helix_n == 2:
        return 12.0
    return None


def _template_contact_reward(coords, template_pairs):
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if not template_pairs:
        return 0.0
    vals = []
    for i, j, d0 in template_pairs:
        if 0 <= i < n and 0 <= j < n and abs(i - j) >= 4:
            d = float(torch.norm(ca[i] - ca[j]).item())
            vals.append(_interval_contact_score(d, good_lo=max(6.8, d0 - 1.5), good_hi=d0 + 1.5, far_cut=d0 + 3.5, clash_lo=5.8))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _core_burial_reward(coords, core_idx):
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if n < 4 or not core_idx:
        return 0.0
    idx = [i for i in core_idx if 0 <= i < n]
    if not idx:
        return 0.0
    center = ca.mean(dim=0, keepdim=True)
    d = torch.norm(ca[idx] - center, dim=-1)
    reward = torch.exp(-(d ** 2) / (2.0 * 3.2 ** 2))
    return float(reward.mean().item())


def _surface_exposure_reward(coords, surface_idx):
    ca = coords[:, 1, :]
    n = ca.shape[0]
    if n < 4 or not surface_idx:
        return 0.0
    idx = [i for i in surface_idx if 0 <= i < n]
    if not idx:
        return 0.0
    center = ca.mean(dim=0, keepdim=True)
    d = torch.norm(ca[idx] - center, dim=-1)
    reward = 1.0 - torch.exp(-(d ** 2) / (2.0 * 4.0 ** 2))
    return float(reward.mean().item())


def _score_with_priors(coords, model, rec_feats, rec_centers, rec_mask, pocket_center,
                       K=50, fps_ratio=0.05, eta=8, proj_iters=60, device=None,
                       w_pocket=0.5, w_compact=0.35, w_clash=2.0, w_step=1.5,
                       w_torsion=0.002, w_helix=0.35, w_contact=1.35, w_template=1.80, w_end=0.08,
                       w_core=0.90, w_surface=0.20,
                       phi_deg=None, psi_deg=None, omega_deg=None,
                       phi_ref=None, psi_ref=None, omega_ref=None,
                       target_end_dist=None, template_pairs=None, seg_meta=None):
    raw = _stage3_score_surface(
        coords, model, rec_feats, rec_centers, rec_mask,
        K=K, fps_ratio=fps_ratio, eta=eta, proj_iters=proj_iters, device=device,
    )
    pocket_pen = _pocket_penalty(coords, pocket_center)
    compact_pen = _compactness_penalty(coords)
    clash_pen = _ca_clash_penalty(coords)
    step_pen = _ca_step_penalty(coords)
    tors_pen = 0.0
    if phi_deg is not None and psi_deg is not None and omega_deg is not None:
        tors_pen = _torsion_prior_penalty(
            phi_deg, psi_deg, omega_deg,
            phi_ref=(-65.0 if phi_ref is None else phi_ref),
            psi_ref=(-40.0 if psi_ref is None else psi_ref),
            omega_ref=(180.0 if omega_ref is None else omega_ref),
        )
    helix_bonus = _helix_reward(coords)
    blueprint = None if seg_meta is None else seg_meta.get("topology_blueprint")
    contact_bonus = _long_range_contact_reward(coords, blueprint=blueprint)
    if template_pairs is None and seg_meta is not None:
        template_pairs = _build_template_contact_pairs(seg_meta)
    template_bonus = _template_contact_reward(coords, template_pairs or [])
    core_bonus = _core_burial_reward(coords, [] if blueprint is None else blueprint.get("core_residues", []))
    surface_bonus = _surface_exposure_reward(coords, [] if blueprint is None else blueprint.get("surface_residues", []))
    end_pen = 0.0
    if target_end_dist is not None:
        end_pen = _end_distance_penalty(coords, target_end_dist=target_end_dist)

    total = raw
    total -= w_pocket * pocket_pen
    total -= w_compact * compact_pen
    total -= w_clash * clash_pen
    total -= w_step * step_pen
    total -= w_torsion * tors_pen
    total += w_helix * helix_bonus
    total += w_contact * contact_bonus
    total += w_template * template_bonus
    total += w_core * core_bonus
    total += w_surface * surface_bonus
    total -= w_end * end_pen

    aux = {
        "raw": raw,
        "pocket_pen": pocket_pen,
        "compact_pen": compact_pen,
        "clash_pen": clash_pen,
        "step_pen": step_pen,
        "torsion_pen": tors_pen,
        "helix_bonus": helix_bonus,
        "contact_bonus": contact_bonus,
        "template_bonus": template_bonus,
        "core_bonus": core_bonus,
        "surface_bonus": surface_bonus,
        "end_pen": end_pen,
        "total": total,
    }
    return total, aux


def optimize_backbone_hillclimb(
    coords_init,
    model,
    rec_feats, rec_centers, rec_mask,
    pocket_center,
    n_steps=30,
    perturb_std=0.3,
    w_pocket=0.5,
    K=50,
    fps_ratio=0.05,
    eta=8,
    proj_iters=60,
    device=None,
    torsion_std=8.0,
    shift_std=0.35,
    omega_std=3.0,
    w_compact=0.35,
    w_clash=2.0,
    w_step=1.5,
    w_torsion=0.002,
    w_helix=0.35,
    w_contact=1.35,
    w_template=1.80,
    w_end=0.08,
    w_core=0.90,
    w_surface=0.20,
    init_schedule=None,
    target_end_dist=None,
):
    """
    v16-topofold-natural:
    1) 使用更明确的 topology blueprint（接触图 + core/surface 先验）
    2) 降低纯 helix 偏置，提高 foldback / packing 偏置
    3) 由贪婪 hill-climb 改为 simulated annealing，允许阶段性跨越局部最优
    """
    if device is None:
        device = coords_init.device

    n_res = coords_init.shape[0]
    n_tors = max(0, n_res - 1)
    first_ca0, forward0, up0 = _seed_frame_from_coords(coords_init)

    if init_schedule is None:
        phi_ref = np.full((n_tors,), -65.0, dtype=np.float32)
        psi_ref = np.full((n_tors,), -40.0, dtype=np.float32)
        omega_ref = np.full((n_tors,), 180.0, dtype=np.float32)
        torsion_std_scale = np.ones((n_tors,), dtype=np.float32)
        seg_meta = {"segments": [("H", 0, n_res)], "residue_types": ["H"] * n_res, "segment_spec": f"H{n_res}"}
        seg_meta["topology_blueprint"] = _build_topology_blueprint(seg_meta)
    else:
        phi_ref = np.asarray(init_schedule["phi_ref"], dtype=np.float32).copy()
        psi_ref = np.asarray(init_schedule["psi_ref"], dtype=np.float32).copy()
        omega_ref = np.asarray(init_schedule["omega_ref"], dtype=np.float32).copy()
        torsion_std_scale = np.asarray(init_schedule.get("torsion_std_scale", np.ones_like(phi_ref)), dtype=np.float32).copy()
        seg_meta = init_schedule.get("seg_meta", {"segments": [], "residue_types": [], "segment_spec": ""})
        if "topology_blueprint" not in seg_meta:
            seg_meta["topology_blueprint"] = _build_topology_blueprint(seg_meta)

    phi_curr = phi_ref.copy()
    psi_curr = psi_ref.copy()
    omega_curr = omega_ref.copy()
    shift_curr = torch.zeros(3, device=device, dtype=coords_init.dtype)
    template_pairs = _build_template_contact_pairs(seg_meta)
    if target_end_dist is None:
        target_end_dist = _default_target_end_dist(seg_meta)

    curr_coords = build_chain_from_torsions(first_ca0 + shift_curr, forward0, up0, phi_curr, psi_curr, omega_curr)
    curr_total, curr_aux = _score_with_priors(
        curr_coords, model, rec_feats, rec_centers, rec_mask, pocket_center,
        K=K, fps_ratio=fps_ratio, eta=eta, proj_iters=proj_iters, device=device,
        w_pocket=w_pocket, w_compact=w_compact, w_clash=w_clash, w_step=w_step,
        w_torsion=w_torsion, w_helix=w_helix, w_contact=w_contact, w_template=w_template, w_end=w_end,
        w_core=w_core, w_surface=w_surface,
        phi_deg=phi_curr, psi_deg=psi_curr, omega_deg=omega_curr,
        phi_ref=phi_ref, psi_ref=psi_ref, omega_ref=omega_ref,
        target_end_dist=target_end_dist, template_pairs=template_pairs, seg_meta=seg_meta,
    )

    best_coords = curr_coords
    best_total = curr_total
    best_aux = curr_aux
    best_score_raw = curr_aux["raw"]
    accepted = 0

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        T = 0.50 * ((0.03 / 0.50) ** frac)

        phi_new = phi_curr.copy()
        psi_new = psi_curr.copy()
        omega_new = omega_curr.copy()
        shift_new = shift_curr.clone()

        if n_tors > 0:
            move_type = np.random.choice(["local", "local", "block", "loop_focus"], p=[0.35, 0.25, 0.25, 0.15])
            if move_type == "local":
                n_edit = max(1, min(n_tors, int(np.ceil(0.18 * n_tors))))
                edit_idx = np.random.choice(n_tors, size=n_edit, replace=False)
            elif move_type == "block":
                span = max(2, min(n_tors, int(np.ceil(0.22 * n_tors))))
                s0 = np.random.randint(0, max(1, n_tors - span + 1))
                edit_idx = np.arange(s0, min(n_tors, s0 + span))
            else:
                loop_pos = [i for i, t in enumerate(seg_meta.get("residue_types", [])[1:]) if t == "L"]
                if loop_pos:
                    n_edit = max(1, min(len(loop_pos), int(np.ceil(0.65 * len(loop_pos)))))
                    edit_idx = np.array(sorted(np.random.choice(loop_pos, size=n_edit, replace=False)), dtype=np.int64)
                else:
                    n_edit = max(1, min(n_tors, int(np.ceil(0.15 * n_tors))))
                    edit_idx = np.random.choice(n_tors, size=n_edit, replace=False)

            scale = torsion_std_scale[edit_idx]
            step_scale = 1.15 - 0.55 * frac
            phi_new[edit_idx] += np.random.randn(len(edit_idx)).astype(np.float32) * torsion_std * scale * step_scale
            psi_new[edit_idx] += np.random.randn(len(edit_idx)).astype(np.float32) * torsion_std * scale * step_scale
            omega_new[edit_idx] += np.random.randn(len(edit_idx)).astype(np.float32) * omega_std * np.clip(scale, 0.8, 1.5) * step_scale

            pull = 0.10 - 0.04 * frac
            phi_new = (1.0 - pull) * phi_new + pull * phi_ref
            psi_new = (1.0 - pull) * psi_new + pull * psi_ref
            omega_new = (1.0 - 0.08) * omega_new + 0.08 * omega_ref

            phi_new = np.clip(phi_new, -170.0, 80.0)
            psi_new = np.clip(psi_new, -180.0, 180.0)
            omega_new = np.clip(omega_new, 150.0, 210.0)

        shift_new = shift_new + torch.randn_like(shift_new) * (shift_std * (1.0 - 0.45 * frac))
        cand_coords = build_chain_from_torsions(first_ca0 + shift_new, forward0, up0, phi_new, psi_new, omega_new)
        cand_total, cand_aux = _score_with_priors(
            cand_coords, model, rec_feats, rec_centers, rec_mask, pocket_center,
            K=K, fps_ratio=fps_ratio, eta=eta, proj_iters=proj_iters, device=device,
            w_pocket=w_pocket, w_compact=w_compact, w_clash=w_clash, w_step=w_step,
            w_torsion=w_torsion, w_helix=w_helix, w_contact=w_contact, w_template=w_template, w_end=w_end,
            w_core=w_core, w_surface=w_surface,
            phi_deg=phi_new, psi_deg=psi_new, omega_deg=omega_new,
            phi_ref=phi_ref, psi_ref=psi_ref, omega_ref=omega_ref,
            target_end_dist=target_end_dist, template_pairs=template_pairs, seg_meta=seg_meta,
        )

        delta = cand_total - curr_total
        accept = False
        if delta >= 0:
            accept = True
        else:
            try:
                p_accept = math.exp(float(delta) / max(float(T), 1e-6))
            except OverflowError:
                p_accept = 0.0
            if np.random.rand() < p_accept:
                accept = True

        if accept:
            accepted += 1
            phi_curr = phi_new
            psi_curr = psi_new
            omega_curr = omega_new
            shift_curr = shift_new
            curr_coords = cand_coords
            curr_total = cand_total
            curr_aux = cand_aux

            if cand_total > best_total:
                best_total = cand_total
                best_aux = cand_aux
                best_score_raw = cand_aux["raw"]
                best_coords = cand_coords

    print("    [OptSummary] "
          f"raw={best_aux['raw']:.4f}, total={best_aux['total']:.4f}, "
          f"pocket={best_aux['pocket_pen']:.4f}, compact={best_aux['compact_pen']:.4f}, "
          f"clash={best_aux['clash_pen']:.4f}, step={best_aux['step_pen']:.4f}, "
          f"torsion={best_aux['torsion_pen']:.2f}, helix={best_aux['helix_bonus']:.4f}, "
          f"contact={best_aux['contact_bonus']:.4f}, template={best_aux['template_bonus']:.4f}, "
          f"core={best_aux['core_bonus']:.4f}, surface={best_aux['surface_bonus']:.4f}, "
          f"end={best_aux['end_pen']:.4f}, accept={accepted}/{max(n_steps,1)}")

    return best_coords, best_score_raw

def _rebuild_fullatom_from_ca(ca_coords, device):
    """
    保留该函数用于兼容旧调用，但实现改为“确定性局部参考系”版本，
    不再为每个残基引入随机 up 向量。
    """
    n_res = ca_coords.shape[0]
    if n_res == 0:
        return torch.empty((0, 4, 3), device=device, dtype=torch.float32)
    eps = 1e-8
    residues = []

    for i in range(n_res):
        ca_i = ca_coords[i]
        if i == 0:
            if n_res > 1:
                fwd = _safe_unit(ca_coords[1] - ca_i)
            else:
                fwd = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=ca_i.dtype)
            if n_res > 2:
                ref = ca_coords[2] - ca_coords[1]
            else:
                ref = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=ca_i.dtype)
            up = _orthogonal_unit(fwd, ref=ref)
        else:
            prev_ca = ca_coords[i - 1]
            fwd = _safe_unit(ca_i - prev_ca)
            if i + 1 < n_res:
                ref = ca_coords[i + 1] - ca_i
            else:
                ref = ca_i - prev_ca
            up = _orthogonal_unit(fwd, ref=ref)
        ang = torch.deg2rad(torch.tensor(111.2, device=device, dtype=ca_i.dtype))
        N_i = ca_i - 1.458 * fwd
        C_i = ca_i + 1.525 * (torch.cos(ang) * fwd + torch.sin(ang) * up)
        O_i = _place_o_from_ncac(N_i, ca_i, C_i)
        residues.append(torch.stack([N_i, ca_i, C_i, O_i], dim=0))

    return torch.stack(residues, dim=0)


# ============================================================
# Section 4: 保存全原子 PDB (N/CA/C/O)
# ============================================================

def save_fullatom_pdb(coords_ncaco, out_pdb, resname="ALA"):
    if isinstance(coords_ncaco, torch.Tensor):
        coords_np = coords_ncaco.detach().cpu().numpy()
    else:
        coords_np = coords_ncaco
    n_res = coords_np.shape[0]
    atom_names = ["N", "CA", "C", "O"]
    st = Structure.Structure("S")
    mdl = Model.Model(0)
    chn = Chain.Chain("A")
    st.add(mdl); mdl.add(chn)
    serial = 1
    for i in range(n_res):
        res = Residue.Residue((" ", i + 1, " "), resname, " ")
        for j, nm in enumerate(atom_names):
            xyz = coords_np[i, j].tolist()
            atm = Atom.Atom(nm, xyz, 1.0, 1.0, " ", nm.ljust(4), serial, nm[0])
            res.add(atm)
            serial += 1
        chn.add(res)
    io = PDBIO()
    io.set_structure(st)
    io.save(out_pdb)


# ============================================================
# Section 5: Backbone fix with Rosetta
# ============================================================

def _fix_backbone_rosetta(poly_ala_pdb, rec_pdb, fixed_pdb, coord_sd=1.0, iters=100):
    if not PYROSETTA_AVAILABLE:
        import shutil; shutil.copy(poly_ala_pdb, fixed_pdb)
        print("  [SKIP] No PyRosetta"); return
    try:
        pyrosetta.init("-mute all -ignore_unrecognized_res true -detect_disulf false -ignore_zero_occupancy false")
    except RuntimeError:
        pass
    rec_pose = pyrosetta.pose_from_pdb(rec_pdb)
    lig_pose = pyrosetta.pose_from_pdb(poly_ala_pdb)
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.append_pose_to_pose(pose, rec_pose, new_chain=True)
    rosetta.core.pose.append_pose_to_pose(pose, lig_pose, new_chain=True)
    lig_chain = pose.num_chains()
    sfxn = pyrosetta.get_fa_scorefxn()
    sc_type = rosetta.core.scoring.ScoreType
    try:
        sfxn.set_weight(getattr(sc_type, "coordinate_constraint"), 1.0)
    except Exception:
        pass
    fn  = rosetta.core.scoring.func.HarmonicFunc(0.0, float(coord_sd))
    ref = rosetta.core.id.AtomID(1, 1)
    for r in range(pose.chain_begin(lig_chain), pose.chain_end(lig_chain) + 1):
        try:
            idx = pose.residue(r).atom_index("CA")
            aid = rosetta.core.id.AtomID(idx, r)
            pose.add_constraint(rosetta.core.scoring.constraints.CoordinateConstraint(
                aid, ref, pose.xyz(aid), fn))
        except Exception:
            continue
    try:
        mm = rosetta.core.kinematics.MoveMap()
        mm.set_bb(False); mm.set_chi(False)
        for r in range(pose.chain_begin(lig_chain), pose.chain_end(lig_chain) + 1):
            mm.set_bb(r, True); mm.set_chi(r, True)
        fr = rosetta.protocols.relax.FastRelax()
        fr.set_scorefxn(sfxn); fr.max_iter(int(iters))
        fr.set_movemap(mm); fr.apply(pose)
        print("  [OK] FastRelax")
    except Exception as e:
        print(f"  [WARN] FastRelax: {e}")
    try:
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover
        mm2 = rosetta.core.kinematics.MoveMap()
        mm2.set_bb(False); mm2.set_chi(False)
        for r in range(pose.chain_begin(lig_chain), pose.chain_end(lig_chain) + 1):
            mm2.set_bb(r, True)
        mv = MinMover(mm2, sfxn, "lbfgs_armijo_nonmonotone", 0.001, True)
        mv.max_iter(300); mv.apply(pose)
        print("  [OK] MinMover")
    except Exception as e:
        print(f"  [WARN] MinMover: {e}")
    try:
        sub = rosetta.core.pose.Pose()
        rosetta.core.pose.append_subpose_to_pose(
            sub, pose, pose.chain_begin(lig_chain), pose.chain_end(lig_chain), True)
        sub.dump_pdb(fixed_pdb)
        print(f"  [OK] Fixed: {fixed_pdb}")
    except Exception:
        import shutil; shutil.copy(poly_ala_pdb, fixed_pdb)


# ============================================================
# Section 6: ProteinMPNN 序列设计
# ============================================================

def _fix_oxygen_pdb(in_pdb, out_pdb):
    import math as _math
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("X", in_pdb)
    for model in st:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ' or 'O' in res:
                    continue
                if not all(a in res for a in ['N', 'CA', 'C']):
                    continue
                N_a  = res['N'].get_vector().get_array()
                CA_a = res['CA'].get_vector().get_array()
                C_a  = res['C'].get_vector().get_array()
                b = CA_a - N_a; c = C_a - CA_a
                bc = c / (np.linalg.norm(c) + 1e-8)
                n_vec = np.cross(b, c)
                n_norm = np.linalg.norm(n_vec)
                if n_norm < 1e-6:
                    o_coord = C_a + bc * 1.229
                else:
                    n_vec = n_vec / n_norm
                    angle = _math.radians(120.8)
                    perp = np.cross(n_vec, bc)
                    perp = perp / (np.linalg.norm(perp) + 1e-8)
                    o_vec = (bc * _math.cos(_math.pi - angle)
                             + perp * _math.sin(_math.pi - angle))
                    o_coord = C_a + o_vec * 1.229
                res.add(Atom.Atom('O', o_coord, 1.0, 1.0, ' ', ' O  ', None, 'O'))
    io = PDBIO(); io.set_structure(st); io.save(out_pdb)




AA3 = {"A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
       "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
       "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
       "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"}

HYDROPHOBIC_AAS = set("AVILMFYWCGP")
CHARGED_AAS = set("DEKR")
NEG_AAS = set("DE")
POS_AAS = set("KR")
POLAR_AAS = set("NQSTHY")


def _safe_frac(num, den):
    return float(num) / float(max(den, 1))


def _shannon_entropy(seq):
    if not seq:
        return 0.0
    counts = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    n = len(seq)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p + 1e-12)
    return h


def _max_repeat_run(seq):
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _low_complexity_windows(seq, window=6, unique_thresh=3):
    if len(seq) < window:
        return 0
    cnt = 0
    for i in range(len(seq) - window + 1):
        if len(set(seq[i:i + window])) <= unique_thresh:
            cnt += 1
    return cnt


def _clip_indices(idxs, n):
    out = []
    for x in idxs or []:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < n:
            out.append(xi)
    return sorted(set(out))


def _fallback_core_surface_from_pdb(lig_pdb):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("L", lig_pdb)
    res_list = [r for r in next(next(iter(st)).get_chains()).get_residues() if not r.id[0].strip() and 'CA' in r]
    if not res_list:
        return [], []
    ca = np.array([r['CA'].coord for r in res_list], dtype=np.float32)
    center = ca.mean(axis=0, keepdims=True)
    d = np.linalg.norm(ca - center, axis=1)
    order = np.argsort(d)
    n = len(res_list)
    n_core = max(1, int(round(0.30 * n)))
    n_surface = max(1, int(round(0.30 * n)))
    core = sorted(order[:n_core].tolist())
    surface = sorted(order[-n_surface:].tolist())
    return core, surface


def _seq_metrics(seq, core_idx=None, surface_idx=None):
    n = max(len(seq), 1)
    core_idx = _clip_indices(core_idx, len(seq))
    surface_idx = _clip_indices(surface_idx, len(seq))
    core_seq = ''.join(seq[i] for i in core_idx)
    surface_seq = ''.join(seq[i] for i in surface_idx)
    fE = _safe_frac(seq.count('E'), n)
    fDE = _safe_frac(sum(aa in NEG_AAS for aa in seq), n)
    fCharge = _safe_frac(sum(aa in CHARGED_AAS for aa in seq), n)
    fHyd = _safe_frac(sum(aa in HYDROPHOBIC_AAS for aa in seq), n)
    fA = _safe_frac(seq.count('A'), n)
    entropy = _shannon_entropy(seq)
    max_run = _max_repeat_run(seq)
    low_complex = _low_complexity_windows(seq, window=6, unique_thresh=3)
    core_hyd = _safe_frac(sum(aa in HYDROPHOBIC_AAS for aa in core_seq), len(core_seq)) if core_seq else fHyd
    core_charge = _safe_frac(sum(aa in CHARGED_AAS for aa in core_seq), len(core_seq)) if core_seq else fCharge
    surface_charge = _safe_frac(sum(aa in CHARGED_AAS or aa in POLAR_AAS for aa in surface_seq), len(surface_seq)) if surface_seq else _safe_frac(sum((aa in CHARGED_AAS) or (aa in POLAR_AAS) for aa in seq), n)
    return {
        'fE': fE, 'fDE': fDE, 'fCharge': fCharge, 'fHyd': fHyd, 'fA': fA,
        'entropy': entropy, 'max_run': max_run, 'low_complex': low_complex,
        'core_hyd': core_hyd, 'core_charge': core_charge, 'surface_charge': surface_charge,
        'n_core': len(core_seq), 'n_surface': len(surface_seq),
    }


def _composition_penalty(metrics):
    pen = 0.0
    pen += 8.0 * max(0.0, metrics['fE'] - 0.18)
    pen += 7.0 * max(0.0, metrics['fDE'] - 0.28)
    pen += 6.0 * max(0.0, metrics['fCharge'] - 0.42)
    pen += 3.0 * max(0.0, metrics['fA'] - 0.22)
    pen += 4.5 * max(0.0, 2.15 - metrics['entropy'])
    pen += 2.0 * max(0, metrics['max_run'] - 3)
    pen += 0.8 * metrics['low_complex']
    pen += 6.0 * max(0.0, 0.42 - metrics['core_hyd'])
    pen += 5.0 * max(0.0, metrics['core_charge'] - 0.25)
    pen += 1.2 * max(0.0, 0.35 - metrics['surface_charge'])
    pen += 1.5 * max(0.0, 0.20 - metrics['fHyd'])
    return pen


def _composition_bonus(metrics):
    bonus = 0.0
    bonus += 1.5 * min(metrics['core_hyd'], 0.70)
    bonus += 0.8 * min(metrics['surface_charge'], 0.75)
    bonus += 0.4 * min(metrics['entropy'] / 2.8, 1.0)
    return bonus


def _seq_rerank_score(seq, nll, core_idx=None, surface_idx=None):
    metrics = _seq_metrics(seq, core_idx=core_idx, surface_idx=surface_idx)
    comp_pen = _composition_penalty(metrics)
    comp_bonus = _composition_bonus(metrics)
    score = float(nll) + comp_pen - comp_bonus
    metrics['comp_pen'] = comp_pen
    metrics['comp_bonus'] = comp_bonus
    metrics['rerank_score'] = score
    return score, metrics


def design_with_proteinmpnn(rec_pdb, lig_pdb, out_pdb,
                            mpnn_samples=16, temperature=0.5, seed=1234, topology_blueprint=None):
    if MPNN_ROOT is None:
        import shutil; shutil.copy(lig_pdb, out_pdb)
        print("  [SKIP] ProteinMPNN not found."); return "A" * 24
    try:
        from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB, _scores
    except ImportError as e:
        print(f"  [ERROR] {e}")
        import shutil; shutil.copy(lig_pdb, out_pdb); return "A" * 24

    import copy
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    wts = [os.path.join(MPNN_ROOT, "vanilla_model_weights", "v_48_002.pt"),
           os.path.join(MPNN_ROOT, "vanilla_model_weights", "v_48_020.pt")]
    wt = next((p for p in wts if os.path.isfile(p)), None)
    if wt is None:
        import shutil; shutil.copy(lig_pdb, out_pdb); return "A" * 24
    print(f"  [MPNN] weights: {os.path.basename(wt)}")
    ck = torch.load(wt, map_location="cpu", weights_only=False)
    mpnn_model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=ck["num_edges"],
    ).to(device).eval()
    mpnn_model.load_state_dict(ck["model_state_dict"])

    rec_list = parse_PDB(rec_pdb, ca_only=False)
    lig_pdb_o = out_pdb + "_tmp_o.pdb"
    _fix_oxygen_pdb(lig_pdb, lig_pdb_o)
    lig_list = parse_PDB(lig_pdb_o, ca_only=False)
    if not rec_list or not lig_list:
        print("  [ERROR] parse_PDB failed")
        import shutil; shutil.copy(lig_pdb, out_pdb); return "A" * 24

    rec_d = rec_list[0]; lig_d = lig_list[0]
    rec_chains = sorted([k.split("seq_chain_")[1] for k in rec_d if k.startswith("seq_chain_")])
    lig_chains_orig = sorted([k.split("seq_chain_")[1] for k in lig_d if k.startswith("seq_chain_")])
    print(f"  [MPNN] Rec chains: {rec_chains}, Lig chains: {lig_chains_orig}")

    used = set(rec_chains)
    lig_rename = {}
    for ch in lig_chains_orig:
        if ch not in used:
            lig_rename[ch] = ch; used.add(ch)
        else:
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if letter not in used:
                    lig_rename[ch] = letter; used.add(letter); break
    lig_chains_new = [lig_rename[ch] for ch in lig_chains_orig]
    print(f"  [MPNN] Lig chains renamed: {lig_rename}")

    combined = copy.deepcopy(rec_d)
    combined["name"] = "complex"
    combined["seq"] += "".join([lig_d[f"seq_chain_{ch}"] for ch in lig_chains_orig])
    for ch_orig, ch_new in lig_rename.items():
        combined[f"seq_chain_{ch_new}"] = lig_d[f"seq_chain_{ch_orig}"]
        old_coords = lig_d[f"coords_chain_{ch_orig}"]
        new_coords = {}
        for atom in ["N", "CA", "C", "O"]:
            old_key = f"{atom}_chain_{ch_orig}"
            new_key = f"{atom}_chain_{ch_new}"
            if old_key in old_coords:
                new_coords[new_key] = old_coords[old_key]
        combined[f"coords_chain_{ch_new}"] = new_coords

    chain_id_dict = {"complex": (lig_chains_new, rec_chains)}
    try:
        (
            X, S, mask, lengths,
            chain_M, chain_encoding_all,
            chain_list_list, visible_list_list, masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos, omit_AA_mask, residue_idx,
            dihedral_mask, tied_pos_list_of_lists_list,
            pssm_coef, pssm_bias, pssm_log_odds_all,
            bias_by_res_all, tied_beta
        ) = tied_featurize(
            [combined], device, chain_id_dict,
            fixed_position_dict=None, omit_AA_dict=None,
            tied_positions_dict=None, pssm_dict=None,
            bias_by_res_dict=None, ca_only=False,
        )
    except Exception as e:
        print(f"  [ERROR] tied_featurize: {e}")
        import shutil; shutil.copy(lig_pdb, out_pdb); return "A" * 24

    n_design = int(chain_M.sum().item())
    print(f"  [MPNN] Total residues: {X.shape[1]}, design positions: {n_design}, T={temperature}")
    omit_AAs_np = np.zeros(21)
    bias_AAs_np = np.zeros(21)
    best_seq = None
    best_nll = 1e9
    best_rerank = 1e9
    best_metrics = None
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    if topology_blueprint is None:
        core_idx, surface_idx = _fallback_core_surface_from_pdb(lig_pdb)
    else:
        core_idx = topology_blueprint.get("core_residues", [])
        surface_idx = topology_blueprint.get("surface_residues", [])
    core_idx = _clip_indices(core_idx, n_design)
    surface_idx = _clip_indices(surface_idx, n_design)
    print(f"  [MPNN] core_idx={len(core_idx)}, surface_idx={len(surface_idx)}")

    with torch.no_grad():
        for s in range(mpnn_samples):
            try:
                randn = torch.randn(chain_M.shape, device=device)
                sample_dict = mpnn_model.sample(
                    X, randn, S, chain_M, chain_encoding_all, residue_idx,
                    mask=mask, temperature=temperature,
                    omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                    chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                    pssm_multi=0.0, pssm_log_odds_flag=False,
                    pssm_log_odds_mask=(pssm_log_odds_all > 0).float(),
                    pssm_bias_flag=False, bias_by_res=bias_by_res_all,
                )
                S_sample = sample_dict["S"]
                design_mask = chain_M[0].bool()
                design_seq = "".join([alphabet[int(i) % 21]
                                      for i in S_sample[0][design_mask].cpu().numpy()])
                log_probs = mpnn_model(X, S_sample, mask, chain_M,
                                      residue_idx, chain_encoding_all, randn)
                score = _scores(S_sample, log_probs, mask * chain_M).item()
                rerank_score, metrics = _seq_rerank_score(design_seq, score, core_idx=core_idx, surface_idx=surface_idx)
                print(
                    f"  [MPNN] Sample {s+1}/{mpnn_samples}: {design_seq[:12]}... "
                    f"NLL={score:.3f} rerank={rerank_score:.3f} E={metrics['fE']:.2f} "
                    f"charge={metrics['fCharge']:.2f} H={metrics['entropy']:.2f} coreHyd={metrics['core_hyd']:.2f}"
                )
                if (not np.isnan(score)) and (not np.isnan(rerank_score)):
                    better = (rerank_score < best_rerank - 1e-8) or (
                        abs(rerank_score - best_rerank) <= 1e-8 and score < best_nll
                    )
                    if better:
                        best_nll = score
                        best_rerank = rerank_score
                        best_seq = design_seq
                        best_metrics = metrics
            except Exception as e:
                print(f"  [WARN] sample {s}: {e}")

    if best_seq is None:
        best_seq = "A" * n_design
        best_metrics = _seq_metrics(best_seq, core_idx=core_idx, surface_idx=surface_idx)
        print("  [WARN] All samples failed, using poly-ALA")
    print(
        f"  [MPNN] Best: {best_seq} (NLL={best_nll:.3f}, rerank={best_rerank:.3f}, "
        f"E={best_metrics['fE']:.2f}, charge={best_metrics['fCharge']:.2f}, "
        f"entropy={best_metrics['entropy']:.2f}, coreHyd={best_metrics['core_hyd']:.2f})"
    )

    # 把设计序列写入配体 PDB
    parser_bio = PDBParser(QUIET=True)
    st = parser_bio.get_structure("L", lig_pdb)
    res_list = [r for r in next(next(iter(st)).get_chains()).get_residues()
                if not r.id[0].strip()]
    for res, aa in zip(res_list, best_seq):
        res.resname = AA3.get(aa, "ALA")
    io = PDBIO(); io.set_structure(st); io.save(out_pdb)
    print(f"  [MPNN] Saved: {out_pdb}")
    return best_seq


# ============================================================
# Section 7: 主生成器
# ============================================================

class FullAtomBackboneGenerator:
    """
    全原子骨架生成器：
      1. NeRF 生长 N/CA/C/O 骨架
      2. 动态生成 surface patch 特征
      3. Stage3 无梯度随机游走优化
      4. 保存最优骨架为 poly-ALA PDB
    """

    def __init__(self, model, device, K=50,
                 phi_noise=15.0, psi_noise=15.0,
                 opt_steps=30, perturb_std=0.3, w_pocket=0.5,
                 fps_ratio=0.05, eta=8, proj_iters=60,
                 torsion_std=8.0, shift_std=0.35, omega_std=3.0,
                 w_compact=0.35, w_clash=2.0, w_step=1.5,
                 w_torsion=0.002, w_helix=0.35,
                 topology_mode="auto", segment_spec=None,
                 w_contact=1.35, w_template=1.80, w_end=0.08,
                 w_core=0.90, w_surface=0.20,
                 target_end_dist=None):
        self.model = model
        self.device = device
        self.K = K
        self.phi_noise = phi_noise
        self.psi_noise = psi_noise
        self.opt_steps = opt_steps
        self.perturb_std = perturb_std
        self.w_pocket = w_pocket
        self.fps_ratio = fps_ratio
        self.eta = eta
        self.proj_iters = proj_iters
        self.torsion_std = torsion_std
        self.shift_std = shift_std
        self.omega_std = omega_std
        self.w_compact = w_compact
        self.w_clash = w_clash
        self.w_step = w_step
        self.w_torsion = w_torsion
        self.w_helix = w_helix
        self.topology_mode = topology_mode
        self.segment_spec = segment_spec
        self.w_contact = w_contact
        self.w_template = w_template
        self.w_end = w_end
        self.w_core = w_core
        self.w_surface = w_surface
        self.target_end_dist = target_end_dist

    @torch.no_grad()
    def _get_pocket_center(self, rec_feats, rec_centers, rec_mask):
        enc = self.model.encoder.rec_encoder(rec_feats, rec_centers)
        pc, _ = self.model.pocket_head(enc, rec_mask)
        return pc[0]  # [3]

    def generate(self, rec_npz, out_pdb, seq_len, n_trials=5):
        print(f"[Gen] {rec_npz}")
        data = np.load(rec_npz, allow_pickle=True)
        n_patch = len(data["patch_centers"])
        idx = np.random.choice(n_patch, min(512, n_patch), replace=False)
        r_knn = data["patch_knn_idx"][idx]
        r_ctr = data["patch_centers"][idx].astype(np.float32)
        r_pts = data["xs"][r_knn].astype(np.float32)
        r_nrm = data["ns"][r_knn].astype(np.float32)
        r_feat = np.concatenate([r_pts - r_ctr[:, None, :], r_nrm], axis=-1)
        rec_feats   = torch.from_numpy(r_feat).unsqueeze(0).to(self.device)
        rec_centers = torch.from_numpy(r_ctr).unsqueeze(0).to(self.device)
        rec_mask    = torch.zeros(1, rec_feats.shape[1], dtype=torch.bool, device=self.device)

        pocket_center = self._get_pocket_center(rec_feats, rec_centers, rec_mask)
        print(f"[Gen] Pocket: {pocket_center.cpu().numpy().round(2)}")
        print(f"[Gen] {seq_len} residues x {n_trials} trials")

        min_len = max(3, int(seq_len * 0.8))
        results = []

        self.last_topology_blueprint = None
        self.last_schedule = None
        for t in range(n_trials):
            # Step 1: NeRF 生长全原子骨架
            coords, schedule = nerf_grow_fullatom(
                seq_len, pocket_center, self.device,
                phi_noise=self.phi_noise, psi_noise=self.psi_noise,
                topology_mode=self.topology_mode,
                segment_spec=self.segment_spec,
            )
            actual_len = coords.shape[0]
            print(f"  Trial {t+1}/{n_trials}: res={actual_len}, topo={schedule.get('segment_spec', 'NA')}, opt {self.opt_steps} steps...")

            if actual_len < min_len:
                # 太短，直接打分不优化
                sc = _stage3_score_surface(
                    coords, self.model, rec_feats, rec_centers, rec_mask,
                    K=self.K, fps_ratio=self.fps_ratio,
                    eta=self.eta, proj_iters=self.proj_iters, device=self.device,
                )
                results.append((coords, sc, actual_len, schedule))
                print(f"  Trial {t+1}/{n_trials}: score={sc:.4f} [no-opt, too short]")
                continue

            # Step 2: 随机游走爬山优化
            best_coords, best_sc = optimize_backbone_hillclimb(
                coords, self.model, rec_feats, rec_centers, rec_mask,
                pocket_center,
                n_steps=self.opt_steps,
                perturb_std=self.perturb_std,
                w_pocket=self.w_pocket,
                K=self.K,
                fps_ratio=self.fps_ratio,
                eta=self.eta,
                proj_iters=self.proj_iters,
                device=self.device,
                torsion_std=self.torsion_std,
                shift_std=self.shift_std,
                omega_std=self.omega_std,
                w_compact=self.w_compact,
                w_clash=self.w_clash,
                w_step=self.w_step,
                w_torsion=self.w_torsion,
                w_helix=self.w_helix,
                w_contact=self.w_contact,
                w_template=self.w_template,
                w_end=self.w_end,
                w_core=self.w_core,
                w_surface=self.w_surface,
                init_schedule=schedule,
                target_end_dist=self.target_end_dist,
            )
            print(f"  Trial {t+1}/{n_trials}: score={best_sc:.4f} [opt]")
            results.append((best_coords, best_sc, actual_len, schedule))

        eligible = [r for r in results if r[2] >= min_len]
        if not eligible:
            eligible = sorted(results, key=lambda x: x[2], reverse=True)[:1]
        best = max(eligible, key=lambda x: x[1])
        best_coords, best_sc, best_len, best_schedule = best
        self.last_schedule = best_schedule
        if isinstance(best_schedule, dict):
            self.last_topology_blueprint = best_schedule.get("topology_blueprint")
        print(f"[Gen] Best: score={best_sc:.4f}, res={best_len}")

        save_fullatom_pdb(best_coords, out_pdb)
        print(f"[Gen] Saved: {out_pdb}")
        return best_coords


# ============================================================
# Section 8: main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage4-v17-natural-diverse: 强化三级结构的分段拓扑 + 稀疏接触图 + 全原子骨架优化")
    parser.add_argument("--rec_npz",  required=True, help="受体 .npz 文件（Stage1 预处理输出）")
    parser.add_argument("--rec_pdb",  required=True, help="受体 PDB 文件（Rosetta 修复和 MPNN 用）")
    parser.add_argument("--ckpt",     required=True, help="Stage3 模型权重 .pt")
    parser.add_argument("--out",      required=True, help="输出 PDB 路径")
    parser.add_argument("--seq_len",  type=int, default=40, help="目标肽链长度")
    parser.add_argument("--n_trials", type=int, default=10, help="骨架生成 trial 数")
    parser.add_argument("--opt_steps",   type=int,   default=30,  help="随机游走优化步数")
    parser.add_argument("--perturb_std", type=float, default=0.3, help="保留旧接口参数；新版优化主用 torsion/shift 扰动")
    parser.add_argument("--w_pocket",    type=float, default=0.5, help="pocket 约束权重")
    parser.add_argument("--phi_noise",   type=float, default=15.0)
    parser.add_argument("--psi_noise",   type=float, default=15.0)
    parser.add_argument("--K",           type=int,   default=50)
    parser.add_argument("--fps_ratio",   type=float, default=0.05, help="surface patch FPS 比例")
    parser.add_argument("--eta",         type=int,   default=8,    help="SDF 采样密度（越大越慢）")
    parser.add_argument("--proj_iters",  type=int,   default=60,   help="SDF 等值面投影迭代次数")
    parser.add_argument("--topology_mode", type=str, default="auto",
                        choices=["auto", "single_helix", "helix_loop_helix", "three_helix", "helix_loop_helix_loop_helix"],
                        help="骨架拓扑模板模式")
    parser.add_argument("--segment_spec", type=str, default="",
                        help="自定义分段模板，如 H10-L4-H11-L4-H10；为空时按 topology_mode 自动生成")
    parser.add_argument("--torsion_std", type=float, default=8.0, help="phi/psi 扰动标准差（度）")
    parser.add_argument("--shift_std",   type=float, default=0.35, help="整条链整体平移扰动（Å）")
    parser.add_argument("--omega_std",   type=float, default=3.0, help="omega 扰动标准差（度）")
    parser.add_argument("--w_compact",   type=float, default=0.35, help="紧致性惩罚权重")
    parser.add_argument("--w_clash",     type=float, default=2.0, help="非局部 CA clash 惩罚权重")
    parser.add_argument("--w_step",      type=float, default=1.5, help="相邻 CA 步长惩罚权重")
    parser.add_argument("--w_torsion",   type=float, default=0.002, help="torsion 偏离 alpha 区域惩罚权重")
    parser.add_argument("--w_helix",     type=float, default=0.35, help="alpha-helix 奖励权重（已降低，避免只长螺旋）")
    parser.add_argument("--w_contact",   type=float, default=1.35, help="长程接触奖励权重")
    parser.add_argument("--w_template",  type=float, default=1.80, help="拓扑模板接触奖励权重")
    parser.add_argument("--w_end",       type=float, default=0.08, help="端到端距离惩罚权重")
    parser.add_argument("--target_end_dist", type=float, default=None, help="目标 N/C 端 CA 距离（Å），为空则不启用端距约束")
    parser.add_argument("--w_core",      type=float, default=0.90, help="核心埋藏奖励权重")
    parser.add_argument("--w_surface",   type=float, default=0.20, help="表面暴露奖励权重")
    parser.add_argument("--mpnn_samples",     type=int,   default=16,  help="ProteinMPNN 采样数")
    parser.add_argument("--mpnn_temperature", type=float, default=0.5)
    parser.add_argument("--fix_backbone", action="store_true", help="是否用 Rosetta 修复骨架")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载 Stage3 模型（先加载到 CPU，再移到目标设备，避免 cuda device index 不匹配）
    ck = torch.load(args.ckpt, map_location='cpu')
    cfg = ck["args"]
    model = DockingModel(
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        nlayers=cfg.get("nlayers", 6),
        dropout=cfg.get("dropout", 0.1),
        K=cfg.get("K", 50),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[Stage3] Loaded: {args.ckpt}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(args.out)[0]
    poly_ala_pdb = base + "_polyALA.pdb"
    fixed_pdb    = base + "_fixed.pdb"

    print("\n=== Step 1: NeRF 全原子骨架生成 + Stage3 优化 ===")
    gen = FullAtomBackboneGenerator(
        model, device, K=args.K,
        phi_noise=args.phi_noise, psi_noise=args.psi_noise,
        opt_steps=args.opt_steps, perturb_std=args.perturb_std,
        w_pocket=args.w_pocket, fps_ratio=args.fps_ratio,
        eta=args.eta, proj_iters=args.proj_iters,
        torsion_std=args.torsion_std, shift_std=args.shift_std, omega_std=args.omega_std,
        w_compact=args.w_compact, w_clash=args.w_clash, w_step=args.w_step,
        w_torsion=args.w_torsion, w_helix=args.w_helix,
        topology_mode=args.topology_mode, segment_spec=(args.segment_spec or None),
        w_contact=args.w_contact, w_template=args.w_template, w_end=args.w_end,
        w_core=args.w_core, w_surface=args.w_surface,
        target_end_dist=args.target_end_dist,
    )
    gen.generate(args.rec_npz, poly_ala_pdb, args.seq_len, n_trials=args.n_trials)

    print("\n=== Step 2: Backbone Fix (Rosetta, 可选) ===")
    if args.fix_backbone:
        _fix_backbone_rosetta(poly_ala_pdb, args.rec_pdb, fixed_pdb)
        input_for_mpnn = fixed_pdb
    else:
        input_for_mpnn = poly_ala_pdb
        print("  [SKIP] --fix_backbone not set")

    print("\n=== Step 3: ProteinMPNN 序列设计 ===")
    best_seq = design_with_proteinmpnn(
        args.rec_pdb, input_for_mpnn, args.out,
        mpnn_samples=args.mpnn_samples,
        temperature=args.mpnn_temperature,
        seed=args.seed,
        topology_blueprint=getattr(gen, "last_topology_blueprint", None),
    )

    print(f"\n=== Done ===")
    print(f"Sequence : {best_seq}")
    print(f"Output   : {args.out}")


if __name__ == "__main__":
    main()
