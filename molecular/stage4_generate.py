from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.train_utils       import set_seed, save_checkpoint, load_checkpoint
from src.datasets_stage4   import LigandGenDataset, gen_collate, load_pdb_for_inference
from src.models            import LigandGenerator
from src.utils             import ELEMENT_LIST, safe_mkdir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Molecule reconstruction helpers
# ---------------------------------------------------------------------------

def _bond_logits_to_edges(
    bond_logits: torch.Tensor,  # (Na, Na, 5)
    positions:   torch.Tensor,  # (Na, 3)
    dist_cutoff: float = 4.5,
) -> tuple:
    """
    Convert bond logits to an edge list, filtering by distance cutoff
    and requiring predicted bond type != 0 (no-bond class).

    Returns
    -------
    edge_index : (2, Nb)  long
    edge_type  : (Nb,)    long  1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC
    """
    Na = positions.shape[0]
    # distance filter
    with torch.no_grad():
        D = torch.cdist(positions.unsqueeze(0),
                        positions.unsqueeze(0)).squeeze(0)  # (Na, Na)
    within = (D < dist_cutoff) & (~torch.eye(Na, dtype=torch.bool,
                                              device=positions.device))
    bond_type = bond_logits.argmax(dim=-1)  # (Na, Na)  0=no-bond
    has_bond  = (bond_type > 0) & within
    # symmetrize: keep upper triangle
    has_bond  = has_bond & has_bond.t()
    src, dst  = has_bond.nonzero(as_tuple=True)
    # keep src < dst (undirected)
    keep      = src < dst
    src, dst  = src[keep], dst[keep]
    btype     = bond_type[src, dst]
    edge_index = torch.stack([src, dst], dim=0)  # (2, Nb)
    return edge_index.cpu(), btype.cpu()


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
        lines.append(f"{a1:3d}{a2:3d}{bt:3s}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("[Stage4] Wrote SDF: %s  (%d atoms, %d bonds)", path, Na, Nb)


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
        ei_list      = [e.to(device) for e in batch["lig_edge_index"]]
        et_list      = [e.to(device) for e in batch["lig_edge_type"]]

        ctx = torch.amp.autocast("cuda") if args.amp else nullcontext()
        with ctx:
            out = model(
                rec_feats, rec_centers, rec_mask,
                lig_pos, lig_atype, lig_mask,
                ei_list, et_list,
                pocket_center=lig_center,
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
            logger.info(
                "Epoch %d | step %d | loss=%.4f coord=%.4f "
                "type=%.4f bond=%.4f natom=%.4f",
                epoch, n, loss.item(),
                out["coord_loss"].item(), out["type_loss"].item(),
                out["bond_loss"].item(), out["natom_loss"].item(),
            )

    return total / max(n, 1)


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
        )

        pos   = out["positions"][0].cpu()    # (Na, 3)
        atype = out["atom_types"][0].cpu()   # (Na,)
        blogs = out["bond_logits"][0].cpu()  # (Na, Na, 5)

        edge_index, edge_type = _bond_logits_to_edges(blogs, pos, dist_cutoff)

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
    p.add_argument("--ckpt",        default="",
                   help="Trained Stage-4 checkpoint (.pt)")
    p.add_argument("--out_dir",     default="./outputs/generated")
    p.add_argument("--n_mols",      type=int,   default=10)
    p.add_argument("--num_atoms",   type=int,   default=None,
                   help="Override atom count (else predicted from pocket)")
    p.add_argument("--temperature", type=float, default=1.0)
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
        if not args.pdb:
            raise ValueError("--pdb is required in generate mode")
        if not args.ckpt or not os.path.isfile(args.ckpt):
            raise ValueError(f"--ckpt must point to a valid checkpoint, got: {args.ckpt}")

        state = torch.load(args.ckpt, map_location=device)
        sd    = state.get("model", state)
        model.load_state_dict(sd, strict=True)
        logger.info("[Stage4] Loaded checkpoint: %s", args.ckpt)

        generate_from_pdb(
            model        = model,
            pdb_path     = args.pdb,
            out_dir      = args.out_dir,
            n_mols       = args.n_mols,
            num_atoms    = args.num_atoms,
            temperature  = args.temperature,
            dist_cutoff  = args.dist_cutoff,
            device       = args.device,
            seq_len      = args.seq_len,
            K            = args.K,
            eta          = args.eta,
            proj_iters   = args.proj_iters,
            target_points= args.target_points,
            fps_ratio    = args.fps_ratio,
            seed         = args.seed,
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

    ds = LigandGenDataset(
        index_file = args.index_file,
        npz_root   = args.npz_root,
        seq_len    = args.seq_len,
        K          = args.K,
        split      = "train",
        seed       = args.seed,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, collate_fn=gen_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer, scaler)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train(model, dl, optimizer, scaler, device, epoch, args)
        scheduler.step()
        logger.info("Epoch %d | avg_loss=%.4f", epoch, avg_loss)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "args": vars(args),
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
