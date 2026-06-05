# -*- coding: utf-8 -*-
"""
Stage-4 dataset utilities.

Two main use-cases:

1. Training the LigandGenerator (supervised, from Stage-1 .npz files)
   -> LigandGenDataset  +  gen_collate

2. Inference on a raw PDB file (no .npz needed)
   -> load_pdb_for_inference()
      Runs Stage-1 preprocessing on-the-fly (surface + patches) and
      returns tensors ready to feed into LigandGenerator.generate().
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import ELEMENT2IDX, ELEMENT_LIST, farthest_point_sampling, knn_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared by training dataset and inference loader
# ---------------------------------------------------------------------------

def _build_rec_feats(
    xs: np.ndarray,           # (Ns, 3)  surface points
    ns: np.ndarray,           # (Ns, 3)  normals
    patch_centers: np.ndarray,# (Nc, 3)
    patch_knn_idx: np.ndarray,# (Nc, K)
    patch_order:   np.ndarray,# (Nc,)    Morton order
    seq_len: int,
    K: int,
    rng: np.random.Generator,
    focus_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample receptor patches and build local surface features.

    If ``focus_center`` is provided, choose the patches nearest to that
    point, then order them by Morton rank.  This keeps Stage-4 training
    conditioned on the ligand pocket instead of a random protein window.

    Returns
    -------
    feats   : (T, K, 6)  float32  [rel_xyz | normals]
    centers : (T, 3)     float32
    """
    Nc = patch_centers.shape[0]
    if Nc <= seq_len:
        sel = patch_order
    elif focus_center is not None:
        fc = np.asarray(focus_center, dtype=np.float32).reshape(1, 3)
        d2 = np.sum((patch_centers - fc) ** 2, axis=1)
        nearest = np.argpartition(d2, seq_len - 1)[:seq_len].astype(np.int64)
        rank = np.empty(Nc, dtype=np.int64)
        rank[patch_order] = np.arange(Nc, dtype=np.int64)
        sel = nearest[np.argsort(rank[nearest])]
    else:
        start = int(rng.integers(0, Nc - seq_len + 1))
        sel   = patch_order[start : start + seq_len]

    pts_idx = patch_knn_idx[sel]          # (T, K)
    ctrs    = patch_centers[sel]          # (T, 3)
    rel_xyz = xs[pts_idx] - ctrs[:, None, :]     # (T, K, 3)
    norms   = ns[pts_idx]                         # (T, K, 3)
    feats   = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)
    return feats, ctrs.astype(np.float32)


def _load_npz_s4(path: str, K: int) -> dict:
    """Load a Stage-1 .npz and return a flat dict of arrays."""
    with np.load(path, allow_pickle=True) as d:
        xs      = d["xs"].astype(np.float32)
        ns      = d["ns"].astype(np.float32)
        centers = d["patch_centers"].astype(np.float32)
        knn     = d["patch_knn_idx"].astype(np.int64)
        order   = d["patch_order"].astype(np.int64)
        meta    = json.loads(str(d["meta"])) if "meta" in d else {}

        def _f(key, dtype):
            return d[key].astype(dtype) if key in d else None

        lig_pos       = _f("lig_pos",        np.float32)
        lig_atom_type = _f("lig_atom_type",  np.int32)
        lig_atom_elem = _f("lig_atom_element", np.int32)
        lig_edge_idx  = _f("lig_edge_index", np.int32)
        lig_edge_type = _f("lig_edge_type",  np.int32)
        lig_center    = _f("lig_center",     np.float32)
        pocket_lp     = _f("pocket_label_patch", np.float32)

    # clip KNN columns to requested K
    K0 = knn.shape[1]
    if K0 < K:
        knn = np.concatenate([knn, np.tile(knn[:, -1:], (1, K - K0))], axis=1)
    elif K0 > K:
        knn = knn[:, :K]

    return dict(
        xs=xs, ns=ns, centers=centers, knn=knn, order=order, meta=meta,
        lig_pos=lig_pos, lig_atom_type=lig_atom_type,
        lig_atom_element=lig_atom_elem,
        lig_edge_index=lig_edge_idx, lig_edge_type=lig_edge_type,
        lig_center=lig_center, pocket_label_patch=pocket_lp,
    )


def _graph_order_ligand(
    lig_pos: np.ndarray,
    lig_atom_type: np.ndarray,
    lig_edge_idx: np.ndarray,
    lig_edge_type: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(lig_pos.shape[0])
    if n <= 1 or lig_edge_idx is None or lig_edge_idx.size == 0:
        return lig_pos, lig_atom_type, lig_edge_idx, lig_edge_type

    adj: List[List[int]] = [[] for _ in range(n)]
    for k in range(lig_edge_idx.shape[1]):
        i, j = int(lig_edge_idx[0, k]), int(lig_edge_idx[1, k])
        if 0 <= i < n and 0 <= j < n and i != j:
            adj[i].append(j)
            adj[j].append(i)
    degree = np.asarray([len(set(v)) for v in adj], dtype=np.int64)
    heavy_rank = np.asarray([6 if int(a) == ELEMENT2IDX.get("C", 0) else 10 for a in lig_atom_type], dtype=np.int64)

    order: List[int] = []
    seen = np.zeros(n, dtype=bool)
    while len(order) < n:
        unseen = np.where(~seen)[0]
        start = int(unseen[np.lexsort((unseen, -heavy_rank[unseen], -degree[unseen]))][0])
        queue = [start]
        seen[start] = True
        for u in queue:
            order.append(u)
            neigh = sorted(set(adj[u]), key=lambda v: (-degree[v], -heavy_rank[v], v))
            for v in neigh:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)

    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[np.asarray(order, dtype=np.int64)] = np.arange(n, dtype=np.int64)
    lig_pos = lig_pos[order]
    lig_atom_type = lig_atom_type[order]
    if lig_edge_idx is not None and lig_edge_idx.size > 0:
        lig_edge_idx = old_to_new[lig_edge_idx.astype(np.int64)].astype(np.int32)
    return lig_pos, lig_atom_type, lig_edge_idx, lig_edge_type


def _bond_order_from_type(bt: int) -> int:
    bt = int(bt)
    if bt <= 0:
        return 1
    return 1 if bt == 4 else max(1, min(bt, 3))


def _connected_components(n: int, edges: List[Tuple[int, int]]) -> int:
    if n <= 0:
        return 0
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        if 0 <= i < n and 0 <= j < n and i != j:
            adj[i].append(j)
            adj[j].append(i)
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        q: deque[int] = deque([start])
        seen[start] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
    return comps


def _fallback_ligand_fragment_stats(
    lig_atom_type: np.ndarray,
    lig_edge_idx: np.ndarray,
    lig_edge_type: np.ndarray,
) -> np.ndarray:
    n = int(lig_atom_type.shape[0]) if lig_atom_type is not None else 0
    if lig_edge_idx is None or lig_edge_idx.size == 0:
        edges: List[Tuple[int, int]] = []
        edge_types = np.zeros(0, dtype=np.int64)
    else:
        edges = []
        seen = set()
        for k in range(lig_edge_idx.shape[1]):
            i, j = int(lig_edge_idx[0, k]), int(lig_edge_idx[1, k])
            if i == j or i < 0 or j < 0 or i >= n or j >= n:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
        edge_types = lig_edge_type.astype(np.int64) if lig_edge_type is not None else np.ones(len(edges), dtype=np.int64)
    comps = _connected_components(n, edges)
    ring_count = max(0, len(edges) - n + comps)
    aromatic_edges = int((edge_types == 3).sum()) if edge_types.size else 0
    aromatic_ring_count = int(min(ring_count, max(0, round(aromatic_edges / 6.0))))
    hetero = int(sum(int(a) in {ELEMENT2IDX.get("N"), ELEMENT2IDX.get("O"), ELEMENT2IDX.get("S"), ELEMENT2IDX.get("P")} for a in lig_atom_type))
    halogen = int(sum(int(a) in {ELEMENT2IDX.get("F"), ELEMENT2IDX.get("CL"), ELEMENT2IDX.get("BR"), ELEMENT2IDX.get("I")} for a in lig_atom_type))
    hetero_ring_count = int(min(ring_count, max(0, round(hetero / 3.0))))
    deg = np.zeros(n, dtype=np.int64)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    substituent_count = int((deg == 1).sum())
    branch_count = int((deg >= 3).sum())
    linker_length = int(max(0, n - min(n, ring_count * 6) - substituent_count))
    carbon = int((lig_atom_type == ELEMENT2IDX.get("C", 0)).sum()) if lig_atom_type is not None else 0
    hetero_ratio_bucket = int(np.clip(round(4.0 * hetero / max(n, 1)), 0, 4))
    scaffold_class = int(np.clip(ring_count, 0, 4))
    if aromatic_ring_count > 0:
        scaffold_class = min(scaffold_class + 4, 8)
    elif hetero_ring_count > 0:
        scaffold_class = min(scaffold_class + 8, 12)
    return np.asarray([
        scaffold_class,
        ring_count,
        aromatic_ring_count,
        hetero_ring_count,
        linker_length,
        substituent_count,
        hetero,
        halogen,
        branch_count,
        carbon,
        hetero_ratio_bucket,
    ], dtype=np.int64)


def _rdkit_ligand_fragment_stats(
    lig_atom_type: np.ndarray,
    lig_edge_idx: np.ndarray,
    lig_edge_type: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return None
    if lig_atom_type is None or lig_atom_type.shape[0] == 0:
        return None
    bond_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for at in lig_atom_type:
        elem = ELEMENT_LIST[int(at)] if int(at) < len(ELEMENT_LIST) else "C"
        elem = "C" if elem == "OTHER" else elem.capitalize()
        rw.AddAtom(Chem.Atom(elem))
    if lig_edge_idx is not None and lig_edge_idx.size > 0:
        seen = set()
        for k in range(lig_edge_idx.shape[1]):
            i, j = int(lig_edge_idx[0, k]), int(lig_edge_idx[1, k])
            if i == j or i < 0 or j < 0 or i >= len(lig_atom_type) or j >= len(lig_atom_type):
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            bt = int(lig_edge_type[k]) if lig_edge_type is not None and k < len(lig_edge_type) else 0
            rw.AddBond(a, b, bond_map.get(bt, Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    ring_count = int(len(atom_rings))
    aromatic_ring_count = int(sum(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) for ring in atom_rings))
    hetero_ring_count = int(sum(any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6, 1) for i in ring) for ring in atom_rings))
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_atoms = int(scaffold.GetNumHeavyAtoms()) if scaffold is not None else 0
    scaffold_rings = int(scaffold.GetRingInfo().NumRings()) if scaffold is not None else 0
    linker_length = int(max(0, scaffold_atoms - 6 * scaffold_rings))
    substituent_count = int(sum(1 for atom in mol.GetAtoms() if atom.GetDegree() == 1 and not atom.IsInRing()))
    hetero = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
    halogen = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (9, 17, 35, 53)))
    branch_count = int(sum(1 for atom in mol.GetAtoms() if atom.GetDegree() >= 3))
    carbon = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6))
    hetero_ratio_bucket = int(np.clip(round(4.0 * hetero / max(mol.GetNumHeavyAtoms(), 1)), 0, 4))
    scaffold_class = int(np.clip(scaffold_rings, 0, 4))
    if aromatic_ring_count > 0:
        scaffold_class = min(scaffold_class + 4, 8)
    elif hetero_ring_count > 0:
        scaffold_class = min(scaffold_class + 8, 12)
    return np.asarray([
        scaffold_class,
        ring_count,
        aromatic_ring_count,
        hetero_ring_count,
        linker_length,
        substituent_count,
        hetero,
        halogen,
        branch_count,
        carbon,
        hetero_ratio_bucket,
    ], dtype=np.int64)


def ligand_fragment_stats(
    lig_atom_type: np.ndarray,
    lig_edge_idx: np.ndarray,
    lig_edge_type: np.ndarray,
) -> np.ndarray:
    stats = _rdkit_ligand_fragment_stats(lig_atom_type, lig_edge_idx, lig_edge_type)
    if stats is not None:
        return stats
    return _fallback_ligand_fragment_stats(lig_atom_type, lig_edge_idx, lig_edge_type)


def scaffold_fingerprint(
    lig_atom_type: np.ndarray,
    lig_edge_idx: np.ndarray,
    lig_edge_type: np.ndarray,
    n_bits: int = 256,
) -> Tuple[np.ndarray, float]:
    """Bemis-Murcko scaffold Morgan fingerprint target for auxiliary supervision.

    Returns a binary vector and a scalar mask.  The mask is 0 when RDKit cannot
    sanitize the molecule or no non-empty scaffold exists, so training does not
    force invalid/all-zero scaffold targets.
    """
    fp = np.zeros(int(n_bits), dtype=np.float32)
    if lig_atom_type is None or lig_atom_type.shape[0] == 0:
        return fp, 0.0
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return fp, 0.0

    bond_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for at in lig_atom_type:
        elem = ELEMENT_LIST[int(at)] if int(at) < len(ELEMENT_LIST) else "C"
        elem = "C" if elem == "OTHER" else elem.capitalize()
        rw.AddAtom(Chem.Atom(elem))
    if lig_edge_idx is not None and lig_edge_idx.size > 0:
        seen = set()
        for k in range(lig_edge_idx.shape[1]):
            i, j = int(lig_edge_idx[0, k]), int(lig_edge_idx[1, k])
            if i == j or i < 0 or j < 0 or i >= len(lig_atom_type) or j >= len(lig_atom_type):
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            bt = int(lig_edge_type[k]) if lig_edge_type is not None and k < len(lig_edge_type) else 0
            rw.AddBond(a, b, bond_map.get(bt, Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return fp, 0.0
    if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
        return fp, 0.0
    bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(scaffold, radius=2, nBits=int(n_bits))
    DataStructs.ConvertToNumpyArray(bitvect, fp)
    return fp.astype(np.float32), 1.0


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class LigandGenDataset(Dataset):
    """
    Supervised training dataset for Stage-4 LigandGenerator.

    Reads Stage-1 .npz files.  Each sample returns:
      rec_feats     : (T, K, 6)
      rec_centers   : (T, 3)
      rec_mask      : (T,)     True = padding
      lig_pos       : (Na, 3)
      lig_atom_type : (Na,)    long  (ELEMENT_LIST indices)
      lig_mask      : (Na,)    True = padding
      lig_edge_index: (2, Nb)
      lig_edge_type : (Nb,)
      pocket_center : (3,)
      affinity      : float
    """

    def __init__(
        self,
        index_file: str,
        npz_root:   str,
        seq_len:    int  = 512,
        K:          int  = 32,
        split:      Optional[str] = None,
        cache:      bool = True,
        seed:       int  = 2024,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.K         = K
        self._cache: Optional[dict] = {} if cache else None
        self._rng_base = np.random.default_rng(seed)

        df = pd.read_csv(index_file)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.samples: List[dict] = []
        missing = 0
        for _, row in df.iterrows():
            cid = str(row["complex_id"])
            npz = os.path.join(npz_root, f"{cid}.npz")
            if not os.path.isfile(npz):
                missing += 1
                continue
            aff = float(row["affinity"]) \
                if "affinity" in row and not pd.isna(row["affinity"]) else 0.0
            self.samples.append(dict(complex_id=cid, npz=npz, affinity=aff))

        logger.info("LigandGenDataset: %d loaded, %d missing npz",
                    len(self.samples), missing)
        if not self.samples:
            raise RuntimeError(
                f"No samples found. Check npz_root={npz_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]
        rng  = np.random.default_rng(self._rng_base.integers(0, 2**31))

        if self._cache is not None and info["npz"] in self._cache:
            arr = self._cache[info["npz"]]
        else:
            arr = _load_npz_s4(info["npz"], self.K)
            if self._cache is not None:
                self._cache[info["npz"]] = arr

        def _lig(key, fb):
            v = arr[key]
            return v if v is not None else fb

        lig_pos       = _lig("lig_pos",       np.zeros((1, 3),  dtype=np.float32))
        # prefer element index (direct ELEMENT_LIST mapping) over rdkit atom type
        lig_atom_elem = _lig("lig_atom_element", None)
        if lig_atom_elem is not None:
            lig_atom_type = lig_atom_elem.astype(np.int32)
        else:
            lig_atom_type = _lig("lig_atom_type", np.zeros(1, dtype=np.int32))
        lig_edge_idx  = _lig("lig_edge_index", np.zeros((2, 0), dtype=np.int32))
        lig_edge_type = _lig("lig_edge_type",  np.zeros(0,      dtype=np.int32))
        lig_center    = _lig("lig_center",     np.zeros(3,      dtype=np.float32))

        rec_feats, rec_centers = _build_rec_feats(
            arr["xs"], arr["ns"],
            arr["centers"], arr["knn"], arr["order"],
            self.seq_len, self.K, rng,
            focus_center=lig_center,
        )

        # Stage-1 RDKit atom order is not semantically aligned across molecules.
        # Reorder each ligand by graph traversal so Direct atom slots see a more
        # consistent molecular path/tree order during supervised training.
        lig_pos, lig_atom_type, lig_edge_idx, lig_edge_type = _graph_order_ligand(
            lig_pos, lig_atom_type, lig_edge_idx, lig_edge_type)
        ligand_stats = ligand_fragment_stats(lig_atom_type, lig_edge_idx, lig_edge_type)
        scaffold_fp, scaffold_fp_mask = scaffold_fingerprint(lig_atom_type, lig_edge_idx, lig_edge_type, n_bits=256)

        # Stage-1 RDKit bond labels are 0=SINGLE,1=DOUBLE,2=TRIPLE,3=AROMATIC.
        # Stage-4 reserves 0 for "no bond", so true bond labels must be shifted.
        if lig_edge_type is not None and lig_edge_type.size > 0:
            lig_edge_type = lig_edge_type.astype(np.int32) + 1

        # Train the diffusion model in a ligand-centred local frame.
        if lig_pos.shape[0] > 0:
            lig_pos = lig_pos - lig_center[None, :]   # (Na, 3) relative coords

        return dict(
            rec_feats     = torch.from_numpy(rec_feats),
            rec_centers   = torch.from_numpy(rec_centers),
            lig_pos       = torch.from_numpy(lig_pos),
            lig_atom_type = torch.from_numpy(lig_atom_type).long(),
            lig_edge_index= torch.from_numpy(lig_edge_idx),
            lig_edge_type = torch.from_numpy(lig_edge_type).long(),
            ligand_stats  = torch.from_numpy(ligand_stats).long(),
            scaffold_fp   = torch.from_numpy(scaffold_fp).float(),
            scaffold_fp_mask = torch.tensor(scaffold_fp_mask, dtype=torch.float32),
            lig_center    = torch.from_numpy(lig_center),
            affinity      = torch.tensor([info["affinity"]], dtype=torch.float32),
            name          = info["complex_id"],
        )


def gen_collate(batch: List[dict]) -> dict:
    """Pad a list of samples into batched tensors."""
    K      = batch[0]["rec_feats"].shape[1]
    Tr_max = max(b["rec_feats"].shape[0] for b in batch)
    Na_max = max(b["lig_pos"].shape[0]   for b in batch)

    rec_feats_l, rec_centers_l, rec_mask_l = [], [], []
    lig_pos_l, lig_type_l, lig_mask_l      = [], [], []
    ei_list, et_list                        = [], []
    lig_ctr_l, aff_l, names                = [], [], []
    stats_l, scaffold_fp_l, scaffold_fp_mask_l = [], [], []

    for b in batch:
        Tr = b["rec_feats"].shape[0]; pr = Tr_max - Tr
        if pr > 0:
            rec_feats_l.append(torch.cat(
                [b["rec_feats"], torch.zeros(pr, K, 6)], 0))
            rec_centers_l.append(torch.cat(
                [b["rec_centers"], torch.zeros(pr, 3)], 0))
            rec_mask_l.append(torch.cat(
                [torch.zeros(Tr, dtype=torch.bool),
                 torch.ones(pr,  dtype=torch.bool)], 0))
        else:
            rec_feats_l.append(b["rec_feats"])
            rec_centers_l.append(b["rec_centers"])
            rec_mask_l.append(torch.zeros(Tr, dtype=torch.bool))

        Na = b["lig_pos"].shape[0]; pl = Na_max - Na
        if pl > 0:
            lig_pos_l.append(torch.cat(
                [b["lig_pos"], torch.zeros(pl, 3)], 0))
            lig_type_l.append(torch.cat(
                [b["lig_atom_type"],
                 torch.zeros(pl, dtype=torch.long)], 0))
            lig_mask_l.append(torch.cat(
                [torch.zeros(Na, dtype=torch.bool),
                 torch.ones(pl,  dtype=torch.bool)], 0))
        else:
            lig_pos_l.append(b["lig_pos"])
            lig_type_l.append(b["lig_atom_type"])
            lig_mask_l.append(torch.zeros(Na, dtype=torch.bool))

        ei_list.append(b["lig_edge_index"])
        et_list.append(b["lig_edge_type"])
        lig_ctr_l.append(b["lig_center"])
        stats_l.append(b["ligand_stats"])
        scaffold_fp_l.append(b["scaffold_fp"])
        scaffold_fp_mask_l.append(b["scaffold_fp_mask"])
        aff_l.append(b["affinity"])
        names.append(b["name"])

    return dict(
        rec_feats      = torch.stack(rec_feats_l),    # (B, Tr, K, 6)
        rec_centers    = torch.stack(rec_centers_l),  # (B, Tr, 3)
        rec_mask       = torch.stack(rec_mask_l),     # (B, Tr)
        lig_pos        = torch.stack(lig_pos_l),      # (B, Na, 3)
        lig_atom_type  = torch.stack(lig_type_l),     # (B, Na)
        lig_mask       = torch.stack(lig_mask_l),     # (B, Na)
        lig_edge_index = ei_list,                     # list of (2, Nb)
        lig_edge_type  = et_list,                     # list of (Nb,)
        lig_center     = torch.stack(lig_ctr_l),      # (B, 3)
        ligand_stats   = torch.stack(stats_l),        # (B, 11)
        scaffold_fp    = torch.stack(scaffold_fp_l),   # (B, 256)
        scaffold_fp_mask = torch.stack(scaffold_fp_mask_l),  # (B,)
        affinity       = torch.cat(aff_l),            # (B,)
        names          = names,
    )


def load_npz_for_inference(
    npz_path: str,
    device: str = "cpu",
    seq_len: int = 512,
    K: int = 32,
    seed: int = 2024,
    pocket_center: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Load a Stage-1 NPZ and return tensors ready for generation.

    Uses ``lig_center`` from the NPZ as the default pocket anchor, which is
    appropriate for reconstruction/sanity tests on known complexes.
    """
    arr = _load_npz_s4(npz_path, K)
    if pocket_center is None:
        if arr["lig_center"] is not None:
            pocket_center_np = arr["lig_center"].astype(np.float32)
        elif arr["lig_pos"] is not None and arr["lig_pos"].shape[0] > 0:
            pocket_center_np = arr["lig_pos"].mean(axis=0).astype(np.float32)
        else:
            pocket_center_np = arr["centers"].mean(axis=0).astype(np.float32)
    else:
        pocket_center_np = np.asarray(pocket_center, dtype=np.float32).reshape(3)

    rng = np.random.default_rng(seed)
    rec_feats, rec_centers = _build_rec_feats(
        arr["xs"], arr["ns"], arr["centers"], arr["knn"], arr["order"],
        seq_len, K, rng, focus_center=pocket_center_np,
    )

    dev = torch.device(device)
    lig_pos_abs = arr["lig_pos"].astype(np.float32) if arr["lig_pos"] is not None else None
    lig_atom = arr["lig_atom_element"] if arr["lig_atom_element"] is not None else arr["lig_atom_type"]
    lig_atom = lig_atom.astype(np.int64) if lig_atom is not None else None
    lig_edge_index = arr["lig_edge_index"].astype(np.int64) if arr["lig_edge_index"] is not None else None
    lig_edge_type = arr["lig_edge_type"].astype(np.int64) + 1 if arr["lig_edge_type"] is not None else None
    ref_ligand_stats = None
    ref_scaffold_fp = None
    ref_scaffold_fp_mask = None
    if lig_atom is not None:
        raw_edge_type = arr["lig_edge_type"].astype(np.int64) if arr["lig_edge_type"] is not None else None
        raw_edge_index = arr["lig_edge_index"].astype(np.int64) if arr["lig_edge_index"] is not None else None
        ref_ligand_stats = ligand_fragment_stats(lig_atom.astype(np.int32), raw_edge_index, raw_edge_type)
        ref_scaffold_fp, ref_scaffold_fp_mask = scaffold_fingerprint(lig_atom.astype(np.int32), raw_edge_index, raw_edge_type, n_bits=256)

    out = dict(
        rec_feats     = torch.from_numpy(rec_feats).unsqueeze(0).to(dev),
        rec_centers   = torch.from_numpy(rec_centers).unsqueeze(0).to(dev),
        rec_mask      = torch.zeros(1, rec_feats.shape[0], dtype=torch.bool, device=dev),
        pocket_center = torch.from_numpy(pocket_center_np).unsqueeze(0).to(dev),
        npz_path      = npz_path,
    )
    if lig_pos_abs is not None:
        out["ref_lig_pos"] = torch.from_numpy(lig_pos_abs).to(dev)
        out["ref_lig_pos_rel"] = torch.from_numpy(lig_pos_abs - pocket_center_np[None, :]).to(dev)
    if lig_atom is not None:
        out["ref_lig_atom_type"] = torch.from_numpy(lig_atom).long().to(dev)
    if lig_edge_index is not None:
        out["ref_lig_edge_index"] = torch.from_numpy(lig_edge_index).long().to(dev)
    if lig_edge_type is not None:
        out["ref_lig_edge_type"] = torch.from_numpy(lig_edge_type).long().to(dev)
    if ref_ligand_stats is not None:
        out["ref_ligand_stats"] = torch.from_numpy(ref_ligand_stats).long().to(dev)
    if ref_scaffold_fp is not None:
        out["ref_scaffold_fp"] = torch.from_numpy(ref_scaffold_fp).float().to(dev)
        out["ref_scaffold_fp_mask"] = torch.tensor(float(ref_scaffold_fp_mask), dtype=torch.float32, device=dev)
    return out


# ---------------------------------------------------------------------------
# Inference loader: raw PDB -> tensors (no .npz needed)
# ---------------------------------------------------------------------------

def load_pdb_for_inference(
    pdb_path:     str,
    device:       str  = "cpu",
    seq_len:      int  = 512,
    K:            int  = 32,
    # surface generation params (mirrors stage1_preprocess defaults)
    eta:          int   = 8,
    sigma_init:   float = 10.0,
    r_level:      float = 1.05,
    proj_iters:   int   = 100,
    proj_lr:      float = 1e-2,
    inner_thresh: float = 0.5,
    target_points:int   = 20000,
    fps_ratio:    float = 0.05,
    zeta:         int   = 16,
    seed:         int   = 2024,
    chain_ids:    Optional[List[str]] = None,
    pocket_center: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """
    Given a raw protein PDB file, run Stage-1 preprocessing on-the-fly
    and return a dict of batched tensors (batch size = 1) ready for
    LigandGenerator.generate().

    Returns
    -------
    dict with:
      rec_feats    : (1, T, K, 6)
      rec_centers  : (1, T, 3)
      rec_mask     : (1, T)  all False (no padding)
      pocket_center: (1, 3)  mean of patch centres
    """
    # lazy imports to avoid hard dependency when only training
    from .protein_parser import parse_protein_pdb
    from .surface        import generate_surface
    from .neighbors      import residue_neighbors
    from .patchify       import build_patches

    logger.info("[Stage4-Inference] Parsing protein: %s", pdb_path)
    atom_pos, atom_sigma, ca_pos, ca_type = parse_protein_pdb(
        pdb_path, chain_ids=chain_ids)

    logger.info("[Stage4-Inference] Generating surface (eta=%d, iters=%d)...",
                eta, proj_iters)
    xs, ns = generate_surface(
        atom_pos, atom_sigma,
        device       = device,
        eta          = eta,
        sigma_init   = sigma_init,
        r_level      = r_level,
        proj_iters   = proj_iters,
        proj_lr      = proj_lr,
        inner_thresh = inner_thresh,
        target_points= target_points,
        seed         = seed,
    )
    logger.info("[Stage4-Inference] Surface: %d points", xs.shape[0])

    logger.info("[Stage4-Inference] Building patches (fps_ratio=%.3f, K=%d)...",
                fps_ratio, K)
    fps_idx, patch_centers, patch_knn_idx, patch_morton, patch_order = \
        build_patches(xs, fps_ratio=fps_ratio, knn_k=K, seed=seed)

    rng = np.random.default_rng(seed)
    focus = np.asarray(pocket_center, dtype=np.float32) if pocket_center is not None else None
    rec_feats, rec_centers = _build_rec_feats(
        xs, ns, patch_centers, patch_knn_idx, patch_order,
        seq_len, K, rng, focus_center=focus,
    )

    # If no pocket anchor is supplied, keep the old fallback for raw-PDB usage.
    # For meaningful generation, pass a ligand/pocket centre or use --npz.
    if pocket_center is None:
        pocket_center = patch_centers.mean(axis=0).astype(np.float32)  # (3,)
    else:
        pocket_center = np.asarray(pocket_center, dtype=np.float32).reshape(3)

    dev = torch.device(device)
    return dict(
        rec_feats     = torch.from_numpy(rec_feats).unsqueeze(0).to(dev),
        rec_centers   = torch.from_numpy(rec_centers).unsqueeze(0).to(dev),
        rec_mask      = torch.zeros(1, rec_feats.shape[0],
                                    dtype=torch.bool, device=dev),
        pocket_center = torch.from_numpy(pocket_center).unsqueeze(0).to(dev),
        pdb_path      = pdb_path,
    )
    