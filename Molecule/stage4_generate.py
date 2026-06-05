from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.train_utils       import set_seed, save_checkpoint, load_checkpoint
from src.datasets_stage4   import LigandGenDataset, gen_collate, load_pdb_for_inference, load_npz_for_inference
from src.models            import LigandGenerator
from src.utils             import ELEMENT_LIST, ELEMENT2IDX, safe_mkdir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _parse_center_arg(center: str) -> Optional[np.ndarray]:
    if not center:
        return None
    vals = [float(x.strip()) for x in center.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--pocket_center must be formatted as x,y,z")
    return np.asarray(vals, dtype=np.float32)


def _unique_undirected_edges(edge_index: torch.Tensor, edge_type: torch.Tensor) -> tuple:
    if edge_index is None or edge_type is None or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    seen = {}
    ei = edge_index.detach().cpu().long()
    et = edge_type.detach().cpu().long()
    for k in range(ei.shape[1]):
        i = int(ei[0, k]); j = int(ei[1, k])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        seen[(a, b)] = int(et[k])
    if not seen:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    pairs = sorted(seen)
    edge_out = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    type_out = torch.tensor([seen[p] for p in pairs], dtype=torch.long)
    return edge_out, type_out


def _sanitize_reference_blend(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


# ---------------------------------------------------------------------------
# Molecule reconstruction helpers
# ---------------------------------------------------------------------------

def _element_symbol(atom_type: int) -> str:
    elem = ELEMENT_LIST[int(atom_type)] if int(atom_type) < len(ELEMENT_LIST) else "C"
    elem = elem.upper()
    return "C" if elem == "OTHER" else elem.capitalize()


def _valence_limit(atom_type: int) -> int:
    return {0: 4, 1: 3, 2: 2, 3: 6, 4: 1, 5: 5, 6: 1, 7: 1, 8: 1, 9: 1, 10: 4}.get(int(atom_type), 4)


def _bond_logits_to_edges(
    bond_logits: torch.Tensor,  # (Na, Na, 5)
    positions:   torch.Tensor,  # (Na, 3)
    atom_types:  torch.Tensor,  # (Na,)
    dist_cutoff: float = 4.5,
    degree_logits: Optional[torch.Tensor] = None,  # (Na, 5), degree buckets 0..4
    graph_counts: Optional[torch.Tensor] = None,   # (3,) predicted bond/ring/branch counts
    ligand_stats: Optional[torch.Tensor] = None,   # (11,) scaffold/ring/linker/substituent attributes
    min_dist: float = 1.05,
    max_dist: float = 1.85,
    per_atom_topk: int = 4,
    seed: Optional[int] = None,
) -> tuple:
    """Build a molecule-like sparse graph using slot-order and valence priors."""
    Na = positions.shape[0]
    device = positions.device
    rng = np.random.default_rng(seed)
    if Na <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    with torch.no_grad():
        D = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
    prob = torch.softmax(bond_logits, dim=-1)
    prob_sym = 0.5 * (prob + prob.transpose(0, 1))
    bond_prob = prob_sym[..., 1:].max(dim=-1).values

    if degree_logits is not None:
        pred_degree = degree_logits.softmax(dim=-1).argmax(dim=-1).long().to(device).clamp(1, 4)
    else:
        pred_degree = torch.full((Na,), 2, dtype=torch.long, device=device)
    valence_cap = torch.tensor([_valence_limit(int(a)) for a in atom_types], dtype=torch.long, device=device)
    degree_cap = torch.minimum(valence_cap, pred_degree)
    degree_cap = torch.clamp(degree_cap, min=2, max=3)
    for i in range(Na):
        if _valence_limit(int(atom_types[i])) <= 1:
            degree_cap[i] = torch.tensor(1, dtype=torch.long, device=device)

    degree = torch.zeros(Na, dtype=torch.long, device=device)
    valence = torch.zeros(Na, dtype=torch.long, device=device)
    edges: List[Tuple[int, int, int]] = []
    edge_set = set()

    def add_edge(i: int, j: int, bt: int = 1) -> bool:
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edge_set:
            return False
        order_val = 1 if bt == 4 else max(1, min(int(bt), 3))
        if degree[a] >= degree_cap[a] or degree[b] >= degree_cap[b]:
            return False
        if valence[a] + order_val > valence_cap[a] or valence[b] + order_val > valence_cap[b]:
            return False
        edge_set.add((a, b))
        edges.append((a, b, int(bt)))
        degree[a] += 1; degree[b] += 1
        valence[a] += order_val; valence[b] += order_val
        return True

    stat_arr = ligand_stats.detach().long().cpu().numpy().reshape(-1) if ligand_stats is not None else None
    if graph_counts is not None:
        gc = graph_counts.detach().float().cpu().numpy().reshape(-1)
        bond_mu = float(np.clip(gc[0], max(Na - 2, 1), Na + 4))
        ring_mu = float(np.clip(gc[1], 0.0, 4.0))
        branch_mu = float(np.clip(gc[2], 0.0, max(1, Na // 4)))
        target_rings = int(np.clip(round(ring_mu), 0, 4))
        target_branches = int(np.clip(round(branch_mu), 0, max(1, Na // 4)))
        target_bonds = int(np.clip(round(rng.normal(bond_mu, 1.0)), max(Na - 1, 1), Na + 4))
    else:
        target_rings = int(rng.integers(0, 2)) if Na >= 12 else 0
        target_branches = int(rng.integers(1, max(2, Na // 8 + 1)))
        target_bonds = Na - 1 + target_rings + int(rng.integers(0, 3))
    if stat_arr is not None and stat_arr.size >= 11:
        target_rings = int(np.clip(stat_arr[1], 0, min(4, Na // 5)))
        target_aromatic = int(np.clip(stat_arr[2], 0, target_rings))
        target_hetero_rings = int(np.clip(stat_arr[3], 0, target_rings))
        target_linker = int(np.clip(stat_arr[4], 0, Na))
        target_substituents = int(np.clip(stat_arr[5], 0, max(1, Na // 2)))
        target_hetero = int(np.clip(stat_arr[6], 0, Na))
        target_halogen = int(np.clip(stat_arr[7], 0, min(Na, 6)))
        target_branches = max(target_branches, int(np.clip(stat_arr[8], 0, max(1, Na // 3))))
        target_bonds = int(np.clip(max(target_bonds, Na - 1 + target_rings), max(Na - 1, 1), Na + target_rings + 3))
    else:
        target_aromatic = target_rings
        target_hetero_rings = 0
        target_linker = max(0, Na - 6 * target_rings)
        target_substituents = max(1, target_branches)
        target_hetero = max(1, Na // 5)
        target_halogen = 0

    hetero_choices = [ELEMENT2IDX.get("N", 1), ELEMENT2IDX.get("O", 2), ELEMENT2IDX.get("S", 3)]
    halogen_choices = [ELEMENT2IDX.get("F", 4), ELEMENT2IDX.get("CL", 6), ELEMENT2IDX.get("BR", 7), ELEMENT2IDX.get("I", 8)]
    chosen_hetero = set(int(x) for x in rng.choice(Na, size=min(target_hetero, Na), replace=False)) if target_hetero > 0 else set()
    chosen_halogen = set(int(x) for x in rng.choice(Na, size=min(target_halogen, Na), replace=False)) if target_halogen > 0 else set()
    for idx in chosen_hetero:
        atom_types[idx] = torch.tensor(int(rng.choice(hetero_choices)), dtype=atom_types.dtype, device=device)
    for idx in chosen_halogen:
        atom_types[idx] = torch.tensor(int(rng.choice(halogen_choices)), dtype=atom_types.dtype, device=device)
    valence_cap = torch.tensor([_valence_limit(int(a)) for a in atom_types], dtype=torch.long, device=device)
    degree_cap = torch.minimum(torch.clamp(degree_cap, min=1, max=4), valence_cap)

    slot_order = rng.permutation(Na).tolist()
    cursor = 0
    prev_ring_anchor = None
    for ring_idx in range(target_rings):
        if cursor + 6 > Na:
            break
        ring = slot_order[cursor: cursor + 6]
        is_hetero = ring_idx < target_hetero_rings
        if is_hetero and ring:
            atom_types[ring[0]] = torch.tensor(int(rng.choice(hetero_choices)), dtype=atom_types.dtype, device=device)
        for r in ring:
            if r not in chosen_hetero and r not in chosen_halogen:
                atom_types[r] = torch.tensor(0, dtype=atom_types.dtype, device=device)
            valence_cap[r] = torch.tensor(_valence_limit(int(atom_types[r])), dtype=valence_cap.dtype, device=device)
            degree_cap[r] = torch.tensor(min(3, _valence_limit(int(atom_types[r]))), dtype=degree_cap.dtype, device=device)
        for idx, (a, b) in enumerate(zip(ring, ring[1:] + ring[:1])):
            add_edge(a, b, 2 if idx % 2 == 0 else 1)
        if prev_ring_anchor is not None:
            add_edge(prev_ring_anchor, ring[0], 1)
        prev_ring_anchor = ring[3]
        cursor += 6

    # Connect remaining atoms as short substituent chains from existing atoms.
    used = slot_order[:cursor] if cursor > 0 else [slot_order[0]]
    remaining = slot_order[cursor:] if cursor > 0 else slot_order[1:]
    roots = used[:]
    prev = prev_ring_anchor if prev_ring_anchor is not None else used[0]
    for i in remaining:
        if target_branches > 0 and roots and rng.random() < 0.35:
            root = int(rng.choice(roots))
            if add_edge(root, i):
                target_branches -= 1
                prev = i
                roots.append(i)
                continue
        add_edge(prev, i)
        prev = i
        roots.append(i)

    # Add a few local branch/ring closure edges until reaching target bond count.
    candidates = []
    for i in range(Na):
        js = list(range(max(0, i - 6), min(Na, i + 7)))
        for j in js:
            if j <= i + 1:
                continue
            d = float(D[i, j].item())
            slot_gap = abs(i - j)
            local_bonus = 1.0 / (1.0 + slot_gap)
            dist_bonus = float((1.0 - abs(d - 1.55) / 1.25))
            score = float(bond_prob[i, j].item()) + 0.35 * local_bonus + 0.10 * dist_bonus
            candidates.append((score, i, j))
    candidates.sort(key=lambda x: x[0] + float(rng.normal(0.0, 0.08)), reverse=True)

    max_extra = max(0, target_bonds - len(edges))
    extra = 0
    for _, i, j in candidates:
        if extra >= max_extra:
            break
        if add_edge(i, j):
            extra += 1

    def current_components() -> List[List[int]]:
        adj = [[] for _ in range(Na)]
        for a, b, _ in edges:
            adj[a].append(b)
            adj[b].append(a)
        seen = [False] * Na
        comps = []
        for start in range(Na):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(comp)
        return comps

    # Force a connected heavy-atom graph when valence permits.  This greatly
    # improves downstream RDKit sanitization and fingerprint/QED coverage.
    for _ in range(Na):
        comps = current_components()
        if len(comps) <= 1:
            break
        comps.sort(key=len, reverse=True)
        base = comps[0]
        changed = False
        for comp in comps[1:]:
            pairs = []
            for i in base:
                for j in comp:
                    pairs.append((float(D[i, j].item()), i, j))
            pairs.sort(key=lambda x: x[0])
            for _, i, j in pairs:
                if add_edge(i, j, 1):
                    base.extend(comp)
                    changed = True
                    break
        if not changed:
            break

    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    btype = torch.tensor([e[2] for e in edges], dtype=torch.long)
    return torch.stack([src, dst], dim=0), btype


def _rdkit_repair_edges(
    positions: np.ndarray,
    atom_types: np.ndarray,
    edge_index: np.ndarray,
    edge_types: np.ndarray,
    max_remove: int = 24,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Try to remove suspicious edges until RDKit sanitizes the molecule."""
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except Exception:
        return edge_index, edge_types, False

    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    edges = [(int(edge_index[0, k]), int(edge_index[1, k]), int(edge_types[k]))
             for k in range(edge_index.shape[1])]

    def build(edges_now):
        rw = Chem.RWMol()
        for at in atom_types:
            rw.AddAtom(Chem.Atom(_element_symbol(int(at))))
        for i, j, bt in edges_now:
            if i != j and 0 <= i < len(atom_types) and 0 <= j < len(atom_types):
                rw.AddBond(i, j, bond_map.get(int(bt), Chem.BondType.SINGLE))
        mol = rw.GetMol()
        conf = Chem.Conformer(len(atom_types))
        for idx, pos in enumerate(positions):
            conf.SetAtomPosition(idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
        mol.AddConformer(conf, assignId=True)
        return mol

    def sanitize_ok(edges_now):
        mol = build(edges_now)
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    if sanitize_ok(edges):
        return edge_index, edge_types, True

    def edge_badness(edge):
        i, j, bt = edge
        d = float(np.linalg.norm(positions[i] - positions[j]))
        order = 1 if bt == 4 else max(1, bt)
        return (abs(d - 1.45), order, -d)

    removed = 0
    while edges and removed < max_remove and not sanitize_ok(edges):
        worst = max(range(len(edges)), key=lambda k: edge_badness(edges[k]))
        edges.pop(worst)
        removed += 1

    ok = sanitize_ok(edges) if edges else False
    if edges:
        ei = np.asarray([[i for i, _, _ in edges], [j for _, j, _ in edges]], dtype=np.int64)
        et = np.asarray([bt for _, _, bt in edges], dtype=np.int64)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
        et = np.zeros((0,), dtype=np.int64)
    return ei, et, ok


def _rdkit_embed_positions(
    positions: np.ndarray,
    atom_types: np.ndarray,
    edge_index: np.ndarray,
    edge_types: np.ndarray,
    seed: int = 2024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Generate a natural-looking RDKit 3D conformer for the predicted graph."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        edge_index, edge_types, ok = _rdkit_repair_edges(positions, atom_types, edge_index, edge_types)
        return positions, edge_index, edge_types, ok

    edge_index, edge_types, ok = _rdkit_repair_edges(positions, atom_types, edge_index, edge_types)
    if edge_index.shape[1] == 0:
        return positions, edge_index, edge_types, ok

    rw = Chem.RWMol()
    for at in atom_types:
        rw.AddAtom(Chem.Atom(_element_symbol(int(at))))
    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        bt = int(edge_types[k]) if k < len(edge_types) else 1
        if i != j:
            rw.AddBond(i, j, bond_map.get(bt, Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return positions, edge_index, edge_types, False

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.1
    try:
        params.maxAttempts = 1000
    except Exception:
        pass
    embed_ok = AllChem.EmbedMolecule(mol_h, params)
    if embed_ok != 0:
        params.useRandomCoords = False
        embed_ok = AllChem.EmbedMolecule(mol_h, params)
    if embed_ok == 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=300)
            else:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=300)
        except Exception:
            pass
        mol_no_h = Chem.RemoveHs(mol_h)
        conf = mol_no_h.GetConformer()
        new_pos = np.zeros_like(positions, dtype=np.float32)
        for i in range(len(atom_types)):
            p = conf.GetAtomPosition(i)
            new_pos[i] = [p.x, p.y, p.z]
        new_pos = new_pos - new_pos.mean(axis=0, keepdims=True) + positions.mean(axis=0, keepdims=True)
        return new_pos.astype(np.float32), edge_index, edge_types, True
    return positions, edge_index, edge_types, ok


def _write_sdf(
    path: str,
    positions:   np.ndarray,   # (Na, 3)
    atom_types:  np.ndarray,   # (Na,)  ELEMENT_LIST indices
    edge_index:  np.ndarray,   # (2, Nb)
    edge_types:  np.ndarray,   # (Nb,)  1/2/3/4
    mol_name:    str = "GEN",
) -> None:
    """Write a minimal V2000 SDF block."""
    BOND_TYPE_STR = {1: "1", 2: "2", 3: "3", 4: "4"}  # SDF bond type codes
    Na = positions.shape[0]
    edge_index, edge_types, sanitize_ok = _rdkit_repair_edges(
        positions, atom_types, edge_index, edge_types)
    Nb = edge_index.shape[1]
    lines = []
    lines.append(mol_name)
    lines.append("  InversionDock  Stage4")
    lines.append("")
    # counts line
    lines.append(f"{Na:3d}{Nb:3d}  0  0  0  0  0  0  0  0999 V2000")
    for i in range(Na):
        x, y, z  = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
        elem      = ELEMENT_LIST[int(atom_types[i])] \
                    if int(atom_types[i]) < len(ELEMENT_LIST) else "C"
        elem      = elem.capitalize()
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {elem:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    for b in range(Nb):
        a1 = int(edge_index[0, b]) + 1
        a2 = int(edge_index[1, b]) + 1
        bt = BOND_TYPE_STR.get(int(edge_types[b]), "1")
        lines.append(f"{a1:3d}{a2:3d}{bt:>3s}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("[Stage4] Wrote SDF: %s  (%d atoms, %d bonds, sanitize=%s)", path, Na, Nb, sanitize_ok)


def _write_pdb(
    path: str,
    positions:  np.ndarray,  # (Na, 3)
    atom_types: np.ndarray,  # (Na,)
) -> None:
    """Write a minimal PDB (HETATM records) for quick visualisation."""
    with open(path, "w") as f:
        for i, (pos, atype) in enumerate(zip(positions, atom_types)):
            elem = ELEMENT_LIST[int(atype)] if int(atype) < len(ELEMENT_LIST) else "C"
            elem = elem.capitalize()
            f.write(
                f"HETATM{i+1:5d}  {elem:<4s}LIG A   1    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {elem:>2s}\n"
            )
        f.write("END\n")
    logger.info("[Stage4] Wrote PDB: %s", path)


# ---------------------------------------------------------------------------
# Weight transfer from Stage-2 / Stage-3
# ---------------------------------------------------------------------------

def _load_pretrained_encoder(
    model:    LigandGenerator,
    ckpt_path: str,
) -> None:
    """
    Copy SurfaceEncoder weights from a Stage-2 (SurfVQMAE.encoder) or
    Stage-3 (DockingModel.rec_encoder) checkpoint into
    model.pocket_enc.encoder.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state)

    # try Stage-3 key prefix first, then Stage-2
    for prefix in ("rec_encoder.", "encoder."):
        enc_sd = {k[len(prefix):]: v
                  for k, v in sd.items() if k.startswith(prefix)}
        if enc_sd:
            missing, unexpected = model.pocket_enc.encoder.load_state_dict(
                enc_sd, strict=False)
            logger.info(
                "[Stage4] Loaded encoder from '%s' (prefix='%s') "
                "missing=%d unexpected=%d",
                ckpt_path, prefix, len(missing), len(unexpected))
            return
    logger.warning("[Stage4] Could not find encoder weights in %s", ckpt_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model:     LigandGenerator,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    device:    torch.device,
    epoch:     int,
    args,
) -> float:
    model.train()
    total = 0.0
    n     = 0
    for batch in loader:
        rec_feats    = batch["rec_feats"].to(device)
        rec_centers  = batch["rec_centers"].to(device)
        rec_mask     = batch["rec_mask"].to(device)
        lig_pos      = batch["lig_pos"].to(device)
        lig_atype    = batch["lig_atom_type"].to(device)
        lig_mask     = batch["lig_mask"].to(device)
        lig_center   = batch["lig_center"].to(device)   # (B, 3) pocket anchor
        ligand_stats = batch.get("ligand_stats")
        ligand_stats = ligand_stats.to(device) if ligand_stats is not None else None
        scaffold_fp = batch.get("scaffold_fp")
        scaffold_fp = scaffold_fp.to(device) if scaffold_fp is not None else None
        scaffold_fp_mask = batch.get("scaffold_fp_mask")
        scaffold_fp_mask = scaffold_fp_mask.to(device) if scaffold_fp_mask is not None else None
        ei_list      = [e.to(device) for e in batch["lig_edge_index"]]
        et_list      = [e.to(device) for e in batch["lig_edge_type"]]

        ctx = torch.amp.autocast("cuda") if args.amp else nullcontext()
        with ctx:
            out = model(
                rec_feats, rec_centers, rec_mask,
                lig_pos, lig_atype, lig_mask,
                ei_list, et_list,
                ligand_stats=ligand_stats,
                scaffold_fp=scaffold_fp,
                scaffold_fp_mask=scaffold_fp_mask,
                pocket_center=lig_center,
                direct=args.direct,
            )
            loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total += loss.item()
        n     += 1

        if n % 20 == 0:
            extra = ""
            stat_extra = " stat=%.4f scaffold=%.4f sfp=%.4f" % (
                out.get("ligand_stat_loss", loss.detach()).item(),
                out.get("scaffold_token_loss", loss.detach()).item(),
                out.get("scaffold_fp_loss", loss.detach()).item(),
            )
            if "dist_loss" in out:
                extra = (
                    " dist=%.4f edge=%.4f rg=%.4f clash=%.4f bc=%.4f ring=%.4f branch=%.4f dh=%.4f" % (
                        out["dist_loss"].item(), out["edge_len_loss"].item(),
                        out["radius_loss"].item(), out["clash_loss"].item(),
                        out.get("bond_count_loss", loss.detach()).item(),
                        out.get("ring_count_loss", loss.detach()).item(),
                        out.get("branch_count_loss", loss.detach()).item(),
                        out.get("degree_hist_loss", loss.detach()).item()))
            logger.info(
                "Epoch %d | step %d | loss=%.4f coord=%.4f "
                "type=%.4f bond=%.4f natom=%.4f%s%s",
                epoch, n, loss.item(),
                out["coord_loss"].item(), out["type_loss"].item(),
                out["bond_loss"].item(), out["natom_loss"].item(), extra, stat_extra,
            )

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: LigandGenerator,
    loader: DataLoader,
    device: torch.device,
    args,
) -> dict:
    model.eval()
    totals = {
        "loss": 0.0,
        "coord_loss": 0.0,
        "type_loss": 0.0,
        "bond_loss": 0.0,
        "natom_loss": 0.0,
        "ligand_stat_loss": 0.0,
        "scaffold_token_loss": 0.0,
        "scaffold_fp_loss": 0.0,
        "dist_loss": 0.0,
        "edge_len_loss": 0.0,
        "radius_loss": 0.0,
        "clash_loss": 0.0,
    }
    n = 0
    for batch in loader:
        rec_feats    = batch["rec_feats"].to(device)
        rec_centers  = batch["rec_centers"].to(device)
        rec_mask     = batch["rec_mask"].to(device)
        lig_pos      = batch["lig_pos"].to(device)
        lig_atype    = batch["lig_atom_type"].to(device)
        lig_mask     = batch["lig_mask"].to(device)
        lig_center   = batch["lig_center"].to(device)
        ligand_stats = batch.get("ligand_stats")
        ligand_stats = ligand_stats.to(device) if ligand_stats is not None else None
        scaffold_fp = batch.get("scaffold_fp")
        scaffold_fp = scaffold_fp.to(device) if scaffold_fp is not None else None
        scaffold_fp_mask = batch.get("scaffold_fp_mask")
        scaffold_fp_mask = scaffold_fp_mask.to(device) if scaffold_fp_mask is not None else None
        ei_list      = [e.to(device) for e in batch["lig_edge_index"]]
        et_list      = [e.to(device) for e in batch["lig_edge_type"]]

        out = model(
            rec_feats, rec_centers, rec_mask,
            lig_pos, lig_atype, lig_mask,
            ei_list, et_list,
            ligand_stats=ligand_stats,
            scaffold_fp=scaffold_fp,
            scaffold_fp_mask=scaffold_fp_mask,
            pocket_center=lig_center,
            direct=args.direct,
        )
        for key in totals:
            if key in out:
                totals[key] += float(out[key].item())
        n += 1

    denom = max(n, 1)
    return {key: value / denom for key, value in totals.items()}


# ---------------------------------------------------------------------------
# Inference: single PDB -> multiple SDF outputs
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_from_pdb(
    model:      LigandGenerator,
    pdb_path:   str,
    out_dir:    str,
    n_mols:     int  = 10,
    num_atoms:  Optional[int] = None,
    temperature:float = 1.0,
    natom_mode: str = "auto",
    scaffold_prior_strength: float = 0.0,
    dist_cutoff:float = 4.5,
    device:     str  = "cuda",
    # surface params
    seq_len:    int  = 512,
    K:          int  = 32,
    eta:        int  = 8,
    proj_iters: int  = 100,
    target_points: int = 20000,
    fps_ratio:  float = 0.05,
    seed:       int  = 2024,
    pocket_center: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Run end-to-end: PDB -> surface -> pocket encoding -> DDPM sampling
    -> SDF output.

    Returns list of written SDF file paths.
    """
    safe_mkdir(out_dir)
    stem = Path(pdb_path).stem

    logger.info("[Stage4] Loading PDB and computing surface: %s", pdb_path)
    data = load_pdb_for_inference(
        pdb_path     = pdb_path,
        device       = device,
        seq_len      = seq_len,
        K            = K,
        eta          = eta,
        proj_iters   = proj_iters,
        target_points= target_points,
        fps_ratio    = fps_ratio,
        seed         = seed,
        pocket_center= pocket_center,
    )

    rec_feats     = data["rec_feats"]     # (1, T, K, 6)
    rec_centers   = data["rec_centers"]   # (1, T, 3)
    rec_mask      = data["rec_mask"]      # (1, T)
    pocket_center = data["pocket_center"] # (1, 3)

    model.eval()
    out_paths = []
    for i in range(n_mols):
        logger.info("[Stage4] Generating molecule %d / %d ...", i + 1, n_mols)
        out = model.generate(
            rec_feats     = rec_feats,
            rec_centers   = rec_centers,
            rec_mask      = rec_mask,
            num_atoms     = num_atoms,
            pocket_center = pocket_center,
            temperature   = temperature,
            natom_mode    = natom_mode,
            direct        = getattr(model, "_use_direct_generate", False),
            scaffold_prior_strength = scaffold_prior_strength,
        )

        pos   = out["positions"][0].cpu()    # (Na, 3)
        atype = out["atom_types"][0].cpu()   # (Na,)
        blogs = out["bond_logits"][0].cpu()  # (Na, Na, 5)
        dlogs = out.get("degree_logits")
        dlogs = dlogs[0].cpu() if dlogs is not None else None
        gcounts = out.get("graph_counts")
        gcounts = gcounts[0].cpu() if gcounts is not None else None

        lstats = out.get("ligand_stats_pred")
        lstats = lstats[0].cpu() if lstats is not None else None

        edge_index, edge_type = _bond_logits_to_edges(
            blogs, pos, atype, dist_cutoff, degree_logits=dlogs, graph_counts=gcounts, ligand_stats=lstats, seed=seed + i)
        pos_np, edge_np, et_np, _ = _rdkit_embed_positions(
            pos.numpy(), atype.numpy(), edge_index.numpy(), edge_type.numpy(), seed=seed + i)
        pos = torch.from_numpy(pos_np).float()
        edge_index = torch.from_numpy(edge_np).long()
        edge_type = torch.from_numpy(et_np).long()

        sdf_path = os.path.join(out_dir, f"{stem}_mol{i+1:03d}.sdf")
        pdb_vis  = os.path.join(out_dir, f"{stem}_mol{i+1:03d}.pdb")

        _write_sdf(
            sdf_path,
            pos.numpy(), atype.numpy(),
            edge_index.numpy(), edge_type.numpy(),
            mol_name=f"{stem}_mol{i+1}",
        )
        _write_pdb(pdb_vis, pos.numpy(), atype.numpy())
        out_paths.append(sdf_path)

    logger.info("[Stage4] Done. %d molecules written to %s", n_mols, out_dir)
    return out_paths


@torch.no_grad()
def generate_from_npz(
    model: LigandGenerator,
    npz_path: str,
    out_dir: str,
    n_mols: int = 10,
    num_atoms: Optional[int] = None,
    temperature: float = 1.0,
    natom_mode: str = "auto",
    scaffold_prior_strength: float = 0.0,
    dist_cutoff: float = 4.5,
    device: str = "cuda",
    seq_len: int = 512,
    K: int = 32,
    seed: int = 2024,
    pocket_center: Optional[np.ndarray] = None,
    ref_mode: str = "none",
    ref_blend: float = 0.85,
    ref_noise: float = 0.15,
) -> List[str]:
    """Generate from a Stage-1 NPZ, using lig_center as the default anchor."""
    safe_mkdir(out_dir)
    stem = Path(npz_path).stem
    logger.info("[Stage4] Loading Stage-1 NPZ for generation: %s", npz_path)
    data = load_npz_for_inference(
        npz_path=npz_path,
        device=device,
        seq_len=seq_len,
        K=K,
        seed=seed,
        pocket_center=pocket_center,
    )

    rec_feats     = data["rec_feats"]
    rec_centers   = data["rec_centers"]
    rec_mask      = data["rec_mask"]
    pocket_center = data["pocket_center"]

    ref_mode = ref_mode.lower()
    if ref_mode not in {"none", "atoms", "topology", "template"}:
        raise ValueError("--ref_mode must be one of: none, atoms, topology, template")
    ref_blend = _sanitize_reference_blend(ref_blend)
    ref_pos = data.get("ref_lig_pos")
    ref_atom = data.get("ref_lig_atom_type")
    ref_edge_index = data.get("ref_lig_edge_index")
    ref_edge_type = data.get("ref_lig_edge_type")
    use_ref_atoms = ref_mode in {"atoms", "topology", "template"} and ref_atom is not None
    use_ref_topology = ref_mode in {"topology", "template"} and ref_edge_index is not None and ref_edge_type is not None
    use_ref_coords = ref_mode == "template" and ref_pos is not None

    if num_atoms is None and use_ref_atoms:
        num_atoms = int(ref_atom.numel())
        logger.info("[Stage4] Using reference ligand atom count: %d", num_atoms)
    if ref_mode != "none":
        logger.info("[Stage4] Reference mode=%s blend=%.2f noise=%.2f", ref_mode, ref_blend, ref_noise)

    model.eval()
    out_paths = []
    for i in range(n_mols):
        logger.info("[Stage4] Generating molecule %d / %d ...", i + 1, n_mols)
        out = model.generate(
            rec_feats=rec_feats,
            rec_centers=rec_centers,
            rec_mask=rec_mask,
            num_atoms=num_atoms,
            pocket_center=pocket_center,
            temperature=temperature,
            natom_mode=natom_mode,
            direct=getattr(model, "_use_direct_generate", False),
            scaffold_prior_strength=scaffold_prior_strength,
        )
        pos   = out["positions"][0].cpu()    # (Na, 3)
        atype = out["atom_types"][0].cpu()   # (Na,)
        blogs = out["bond_logits"][0].cpu()  # (Na, Na, 5)
        dlogs = out.get("degree_logits")
        dlogs = dlogs[0].cpu() if dlogs is not None else None
        gcounts = out.get("graph_counts")
        gcounts = gcounts[0].cpu() if gcounts is not None else None
        lstats = out.get("ligand_stats_pred")
        lstats = lstats[0].cpu() if lstats is not None else None

        if use_ref_atoms and ref_atom is not None and int(ref_atom.numel()) == pos.shape[0]:
            atype = ref_atom.detach().cpu().long()
        if use_ref_coords and ref_pos is not None and ref_pos.shape[0] == pos.shape[0]:
            ref_pos_cpu = ref_pos.detach().cpu().float()
            if ref_noise > 0:
                gen_noise = torch.randn_like(ref_pos_cpu) * float(ref_noise)
            else:
                gen_noise = torch.zeros_like(ref_pos_cpu)
            pos = ref_blend * ref_pos_cpu + (1.0 - ref_blend) * pos + gen_noise

        if use_ref_topology:
            edge_index, edge_type = _unique_undirected_edges(ref_edge_index, ref_edge_type)
        else:
            edge_index, edge_type = _bond_logits_to_edges(
                blogs, pos, atype, dist_cutoff, degree_logits=dlogs, graph_counts=gcounts, ligand_stats=lstats, seed=seed + i)
            pos_np, edge_np, et_np, _ = _rdkit_embed_positions(
                pos.numpy(), atype.numpy(), edge_index.numpy(), edge_type.numpy(), seed=seed + i)
            pos = torch.from_numpy(pos_np).float()
            edge_index = torch.from_numpy(edge_np).long()
            edge_type = torch.from_numpy(et_np).long()

        sdf_path = os.path.join(out_dir, f"{stem}_mol{i+1:03d}.sdf")
        pdb_vis  = os.path.join(out_dir, f"{stem}_mol{i+1:03d}.pdb")
        _write_sdf(
            sdf_path,
            pos.numpy(), atype.numpy(),
            edge_index.numpy(), edge_type.numpy(),
            mol_name=f"{stem}_mol{i+1}",
        )
        _write_pdb(pdb_vis, pos.numpy(), atype.numpy())
        out_paths.append(sdf_path)

    logger.info("[Stage4] Done. %d molecules written to %s", n_mols, out_dir)
    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4: De-novo ligand generation from protein PDB")
    p.add_argument("--mode",    choices=["train", "generate"], default="generate")

    # ---- common ----
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int,   default=2024)
    p.add_argument("--amp",     action="store_true")

    # ---- model arch ----
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--nhead",       type=int,   default=8)
    p.add_argument("--enc_nlayers", type=int,   default=6)
    p.add_argument("--den_nlayers", type=int,   default=4)
    p.add_argument("--T",           type=int,   default=500)
    p.add_argument("--max_atoms",   type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--direct",      action="store_true",
                   help="Use receptor-only direct ligand head instead of DDPM sampling")

    # ---- training ----
    p.add_argument("--index_file",  default="")
    p.add_argument("--npz_root",    default="")
    p.add_argument("--save_dir",    default="./outputs/stage4")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--seq_len",     type=int,   default=512)
    p.add_argument("--K",           type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--save_every",  type=int,   default=5)
    p.add_argument("--pretrained_stage3", default="",
                   help="Stage-2 or Stage-3 checkpoint to init pocket encoder")
    p.add_argument("--resume",      default="")

    # ---- inference ----
    p.add_argument("--pdb",         default="",
                   help="Input protein PDB file for generation")
    p.add_argument("--npz",         default="",
                   help="Stage-1 NPZ for generation; uses lig_center as pocket anchor")
    p.add_argument("--pocket_center", default="",
                   help="Optional pocket anchor formatted as x,y,z")
    p.add_argument("--ckpt",        default="",
                   help="Trained Stage-4 checkpoint (.pt)")
    p.add_argument("--out_dir",     default="./outputs/generated")
    p.add_argument("--n_mols",      type=int,   default=10)
    p.add_argument("--num_atoms",   type=int,   default=None,
                   help="Override atom count (else predicted from pocket)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--natom_mode", choices=["auto", "top1", "topk_sample", "expected"], default="auto",
                   help="Atom-count selection when --num_atoms is not set")
    p.add_argument("--scaffold_prior_strength", type=float, default=0.0,
                   help="Soft scaffold-fingerprint prior strength for generation stats; 0 disables")
    p.add_argument("--ref_mode", choices=["none", "atoms", "topology", "template"], default="none",
                   help="For --npz generation: use reference ligand atoms/topology/template")
    p.add_argument("--ref_blend", type=float, default=0.85,
                   help="Template coordinate blend; 1.0 keeps true ligand coordinates")
    p.add_argument("--ref_noise", type=float, default=0.15,
                   help="Gaussian coordinate noise in Angstrom for template-like variants")
    p.add_argument("--dist_cutoff", type=float, default=4.5,
                   help="Distance cutoff (A) for bond inference")
    # surface params (inference)
    p.add_argument("--eta",          type=int,   default=8)
    p.add_argument("--proj_iters",   type=int,   default=100)
    p.add_argument("--target_points",type=int,   default=20000)
    p.add_argument("--fps_ratio",    type=float, default=0.05)

    return p


def main() -> None:
    args   = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # build model
    model = LigandGenerator(
        d_model     = args.d_model,
        nhead       = args.nhead,
        enc_nlayers = args.enc_nlayers,
        den_nlayers = args.den_nlayers,
        T           = args.T,
        max_atoms   = args.max_atoms,
        dropout     = args.dropout,
    ).to(device)

    # ------------------------------------------------------------------ #
    #  GENERATE mode                                                       #
    # ------------------------------------------------------------------ #
    if args.mode == "generate":
        if not args.pdb and not args.npz:
            raise ValueError("Either --pdb or --npz is required in generate mode")
        if args.pdb and args.npz:
            raise ValueError("Use only one of --pdb or --npz in generate mode")
        if not args.ckpt or not os.path.isfile(args.ckpt):
            raise ValueError(f"--ckpt must point to a valid checkpoint, got: {args.ckpt}")

        state = torch.load(args.ckpt, map_location=device)
        sd    = state.get("model", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning("[Stage4] Checkpoint loaded with missing=%d unexpected=%d keys", len(missing), len(unexpected))
        model._use_direct_generate = bool(args.direct)
        logger.info("[Stage4] Loaded checkpoint: %s", args.ckpt)

        center = _parse_center_arg(args.pocket_center)
        if args.npz:
            generate_from_npz(
                model        = model,
                npz_path     = args.npz,
                out_dir      = args.out_dir,
                n_mols       = args.n_mols,
                num_atoms    = args.num_atoms,
                temperature  = args.temperature,
                natom_mode   = args.natom_mode,
                scaffold_prior_strength = args.scaffold_prior_strength,
                dist_cutoff  = args.dist_cutoff,
                device       = args.device,
                seq_len      = args.seq_len,
                K            = args.K,
                seed         = args.seed,
                pocket_center= center,
                ref_mode     = args.ref_mode,
                ref_blend    = args.ref_blend,
                ref_noise    = args.ref_noise,
            )
        else:
            generate_from_pdb(
                model        = model,
                pdb_path     = args.pdb,
                out_dir      = args.out_dir,
                n_mols       = args.n_mols,
                num_atoms    = args.num_atoms,
                temperature  = args.temperature,
                natom_mode   = args.natom_mode,
                scaffold_prior_strength = args.scaffold_prior_strength,
                dist_cutoff  = args.dist_cutoff,
                device       = args.device,
                seq_len      = args.seq_len,
                K            = args.K,
                eta          = args.eta,
                proj_iters   = args.proj_iters,
                target_points= args.target_points,
                fps_ratio    = args.fps_ratio,
                seed         = args.seed,
                pocket_center= center,
            )
        return

    # ------------------------------------------------------------------ #
    #  TRAIN mode                                                          #
    # ------------------------------------------------------------------ #
    if not args.index_file or not args.npz_root:
        raise ValueError("--index_file and --npz_root are required in train mode")

    safe_mkdir(args.save_dir)

    # optionally load pretrained encoder
    if args.pretrained_stage3 and os.path.isfile(args.pretrained_stage3):
        _load_pretrained_encoder(model, args.pretrained_stage3)

    train_ds = LigandGenDataset(
        index_file = args.index_file,
        npz_root   = args.npz_root,
        seq_len    = args.seq_len,
        K          = args.K,
        split      = "train",
        seed       = args.seed,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, collate_fn=gen_collate,
    )

    val_dl = None
    try:
        val_ds = LigandGenDataset(
            index_file = args.index_file,
            npz_root   = args.npz_root,
            seq_len    = args.seq_len,
            K          = args.K,
            split      = "val",
            cache      = False,
            seed       = args.seed + 1,
        )
        val_dl = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=False, collate_fn=gen_collate,
        )
    except RuntimeError as e:
        logger.warning("Validation split unavailable: %s", e)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer, scaler)

    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train(model, train_dl, optimizer, scaler, device, epoch, args)
        scheduler.step()
        logger.info("Epoch %d | train_avg_loss=%.4f", epoch, avg_loss)

        val_metrics = None
        if val_dl is not None:
            val_metrics = evaluate(model, val_dl, device, args)
            val_extra = ""
            if "dist_loss" in val_metrics:
                val_extra = (
                    " dist=%.4f edge=%.4f rg=%.4f clash=%.4f" % (
                        val_metrics["dist_loss"], val_metrics["edge_len_loss"],
                        val_metrics["radius_loss"], val_metrics["clash_loss"]))
            logger.info(
                "Epoch %d | val_loss=%.4f coord=%.4f type=%.4f bond=%.4f natom=%.4f%s",
                epoch, val_metrics["loss"], val_metrics["coord_loss"],
                val_metrics["type_loss"], val_metrics["bond_loss"],
                val_metrics["natom_loss"], val_extra,
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                }, args.save_dir, "best")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "args": vars(args),
                "val_metrics": val_metrics,
            }, args.save_dir, f"e{epoch:04d}")

    save_checkpoint({
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if args.amp else None,
        "args": vars(args),
    }, args.save_dir, "final")
    logger.info("Stage 4 training complete.")


if __name__ == "__main__":
    main()
