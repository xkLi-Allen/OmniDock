# -*- coding: utf-8 -*-
"""
Stage-1 for SKEMPI: per-(PDB, chain_group) surface preprocessing.

目标：
  - 对 SKEMPI v2 中的每个 (#Pdb) 的受体链组 / 配体链组，分别生成一个 .npz：
        {pdb_id}_{chains}.npz
    例如: 1A4Y_A.npz, 1A4Y_B.npz, 3HFM_HL.npz, 3HFM_Y.npz

  - 每个 .npz 只包含指定链组上的原子和残基（不会把整复合物混在一起），
    格式与原 data_preprocessing.py 完全一致：
      xs, ns, ca_pos, ca_type, geo_nei_idx, geo_nei_dist, geo_nei_type,
      patch_centers, patch_knn_idx, patch_morton, patch_order, fps_idx, meta

依赖：
  - 当前目录下的 data_preprocessing.py
  - Bio.PDB, pandas

用法示例：
  python stage1_skempi_per_chain.py \
    --skempi_csv /path/to/skempi_v2.csv \
    --pdb_dir   /path/to/skempi_pdbs \
    --out_dir   /path/to/Processed_data_skempi_v2_per_chain \
    --device cuda:0 \
    --eta 20 --sigma_init 10.0 --target_points 50000 --fps_ratio 0.05 \
    --knn_k 50 --zeta 16
"""

import os
import math
import json
import argparse

import numpy as np
import torch
import pandas as pd

from Bio.PDB import PDBParser, is_aa

# 这里直接复用你原始 Stage-1 里的所有几何方法和配置
from data_preprocessing import (
    to_tensor,
    SurfaceSDF,
    project_to_levelset_gd,
    sdf_normals,
    remove_inner_points,
    residue_neighbors,
    farthest_point_sampling,
    knn_indices,
    morton3D,
    safe_mkdir,
    is_processed_ok,
    VDW_SIGMA,
    RES2IDX,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ----------------------------
# 按链组解析 PDB
# ----------------------------

def parse_pdb_atoms_residues_for_chains(pdb_path: str, chain_group: str):
    """
    从 PDB 中只取指定链组上的原子 / 残基。

    chain_group:
      - 类似 'A'、'B'、'HL'、'ABC' 等，表示要保留的链编号集合。
      - 我们简单地把字符串里的每个字符当作一个链 ID：
            'HL' -> {'H','L'}

    返回：
      atom_pos: (Na,3) float32
      atom_sigma: (Na,) float32
      residue_ca_pos: (Nr,3) float32
      residue_type_idx: (Nr,) int64
      backbone_ncac: (Nr,3,3) float32  -- N/CA/C 坐标（NEW）
      bb_valid_mask: (Nr,) bool         -- 该残基是否有完整 N/CA/C（NEW）
    """
    chain_set = set(chain_group)  # 'HL' -> {'H','L'}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    atom_pos = []
    atom_sigma = []
    residue_ca_pos = []
    residue_type_idx = []
    backbone_ncac = []    # NEW: N, CA, C for each residue
    bb_valid_mask = []    # NEW: True if residue has all of N, CA, C

    for model in structure:
        for chain in model:
            cid = str(chain.id).strip()
            if cid not in chain_set:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                resname = res.get_resname()
                idx = RES2IDX.get(resname, -1)
                if "CA" in res:
                    residue_ca_pos.append(res["CA"].get_coord().astype(np.float32))
                    residue_type_idx.append(idx)

                    # NEW: extract N, CA, C backbone atoms
                    has_n  = "N"  in res
                    has_ca = "CA" in res
                    has_c  = "C"  in res
                    if has_n and has_ca and has_c:
                        n_coord  = res["N"].get_coord().astype(np.float32)
                        ca_coord = res["CA"].get_coord().astype(np.float32)
                        c_coord  = res["C"].get_coord().astype(np.float32)
                        backbone_ncac.append([n_coord, ca_coord, c_coord])
                        bb_valid_mask.append(True)
                    else:
                        # Use CA position as fallback for missing atoms
                        ca_coord = res["CA"].get_coord().astype(np.float32)
                        backbone_ncac.append([ca_coord, ca_coord, ca_coord])
                        bb_valid_mask.append(False)

                for atom in res.get_atoms():
                    elem = atom.element.strip().title()
                    if elem == "" or elem not in VDW_SIGMA:
                        continue
                    xyz = atom.get_coord().astype(np.float32)
                    atom_pos.append(xyz)
                    atom_sigma.append(VDW_SIGMA[elem])

    if len(atom_pos) == 0:
        raise ValueError(
            f"No valid atoms parsed in {pdb_path} for chains '{chain_group}'"
        )

    atom_pos = np.asarray(atom_pos, dtype=np.float32)
    atom_sigma = np.asarray(atom_sigma, dtype=np.float32)
    if len(residue_ca_pos) == 0:
        residue_ca_pos = atom_pos.copy()
        residue_type_idx = np.full((residue_ca_pos.shape[0],), -1, dtype=np.int64)
        backbone_ncac = np.zeros((residue_ca_pos.shape[0], 3, 3), dtype=np.float32)
        bb_valid_mask = np.zeros(residue_ca_pos.shape[0], dtype=bool)
    else:
        residue_ca_pos = np.asarray(residue_ca_pos, dtype=np.float32)
        residue_type_idx = np.asarray(residue_type_idx, dtype=np.int64)
        backbone_ncac = np.asarray(backbone_ncac, dtype=np.float32)  # (Nr,3,3)
        bb_valid_mask = np.asarray(bb_valid_mask, dtype=bool)         # (Nr,)

    return atom_pos, atom_sigma, residue_ca_pos, residue_type_idx, backbone_ncac, bb_valid_mask


# ----------------------------
# 单个 (pdb_id, chain_group) 的完整 Surface-VQMAE pipeline
# ----------------------------

def process_one_pdb_chain_group(
    pdb_dir: str,
    out_dir: str,
    pdb_id: str,
    chain_group: str,
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
    """
    对指定 pdb_id 的某个 chain_group（比如 'A' 或 'HL'）生成一个 .npz。
    输出文件名：{pdb_id}_{chain_group}.npz
    """
    name = f"{pdb_id}_{chain_group}"
    out_path = os.path.join(out_dir, f"{name}.npz")

    if (not overwrite) and is_processed_ok(out_path):
        print(f"[SKIP] Exists & valid -> {out_path}")
        return

    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.isfile(pdb_path):
        print(f"[WARN] PDB not found for {pdb_id}, skip {name}")
        return

    print(f"[{name}] Parsing PDB ({pdb_path}) ...")
    atom_pos_np, atom_sig_np, ca_pos_np, ca_type_np, backbone_ncac_np, bb_valid_np = \
        parse_pdb_atoms_residues_for_chains(pdb_path, chain_group)

    # torch buffers
    A = to_tensor(atom_pos_np, device)
    SIG = to_tensor(atom_sig_np, device)

    # 1) sampling around atoms
    print(f"[{name}] Sampling η={eta} around {A.shape[0]} atoms ...")
    centers = np.repeat(atom_pos_np, eta, axis=0)
    noise = np.random.normal(loc=0.0, scale=sigma_init, size=centers.shape).astype(np.float32)
    X0_np = centers + noise
    X0 = to_tensor(X0_np, device=device)

    # 2) SDF projection
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
        X_np, ca_pos_np.astype(np.float32), ca_type_np.astype(np.int16), zeta=zeta,
        desc=f"[{name}] Residue NN"
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
        pdb_id=pdb_id,
        chains=chain_group,
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
        backbone_ncac=backbone_ncac_np,       # NEW: (Nr,3,3) N/CA/C coords
        bb_valid_mask=bb_valid_np,             # NEW: (Nr,) bool mask
        meta=json.dumps(meta),
    )


# ----------------------------
# 主程序：从 skempi_v2.csv 读出所有 (pdb_id, 链组)，统一跑一遍
# ----------------------------

def parse_pdb_field(pdb_str: str):
    """
    把 SKEMPI 的 #Pdb 字段拆成 (pdb_id, rec_chains, lig_chains)

    例如：
      '1A4Y_A_B'     -> '1A4Y', 'A',  'B'
      '1CSE_E_I'     -> '1CSE', 'E',  'I'
      '3HFM_HL_Y'    -> '3HFM', 'HL', 'Y'
      '1OGA_ABC_DE'  -> '1OGA', 'ABC','DE'
    """
    parts = str(pdb_str).split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected #Pdb format: {pdb_str}")
    pdb_id = parts[0]
    rec_chains = parts[1]
    lig_chains = parts[2]
    return pdb_id, rec_chains, lig_chains


def main():
    ap = argparse.ArgumentParser(description="Stage-1 per-chain preprocessing for SKEMPI v2")
    ap.add_argument("--skempi_csv", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/skempi_v2.csv",
                    help="Path to skempi_v2.csv")
    ap.add_argument("--pdb_dir", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/PPB-Affinity-DataPrepWorkflow-main/source_data/SKEMPI v2.0/PDBs",
                    help="Directory containing {pdb_id}.pdb for SKEMPI structures")
    ap.add_argument("--out_dir", type=str, default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/Processed_skempi_per_chain",
                    help="Where to save {pdb_id}_{chains}.npz")

    ap.add_argument("--eta", type=int, default=8)
    ap.add_argument("--sigma_init", type=float, default=10.0)
    ap.add_argument("--r_level", type=float, default=1.05)
    ap.add_argument("--proj_iters", type=int, default=80)
    ap.add_argument("--proj_lr", type=float, default=1e-2)
    ap.add_argument("--inner_thresh", type=float, default=0.5)
    ap.add_argument("--target_points", type=int, default=10000)
    ap.add_argument("--fps_ratio", type=float, default=0.05)
    ap.add_argument("--knn_k", type=int, default=32)
    ap.add_argument("--zeta", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess even if output .npz exists")
    args = ap.parse_args()

    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.skempi_csv, sep=";")
    # 只用 #Pdb 列，不对 affinity 做过滤，预处理可以多做一点
    unique_keys = set()
    for _, row in df.iterrows():
        pdb_str = row["#Pdb"]
        try:
            pdb_id, rec_chains, lig_chains = parse_pdb_field(pdb_str)
        except Exception:
            continue
        unique_keys.add((pdb_id, rec_chains))
        unique_keys.add((pdb_id, lig_chains))

    unique_keys = sorted(unique_keys)
    print(f"[Stage1-SKEMPI] Unique (pdb_id, chains) to process = {len(unique_keys)}")

    for pdb_id, chains in tqdm(unique_keys, desc="Per-chain surfaces"):
        try:
            process_one_pdb_chain_group(
                pdb_dir=args.pdb_dir,
                out_dir=args.out_dir,
                pdb_id=pdb_id,
                chain_group=chains,
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
            print(f"[WARN] Failed on {pdb_id}_{chains}: {e}")

    print("[Stage1-SKEMPI] Done.")


if __name__ == "__main__":
    main()
