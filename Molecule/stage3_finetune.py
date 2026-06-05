# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.train_utils     import set_seed, save_checkpoint, load_checkpoint, load_stage2_encoder
from src.datasets_stage3 import ProteinLigandDataset, finetune_collate
from src.losses          import (pocket_bce_loss, binding_bce_loss,
                                 affinity_loss, contact_loss, flex_loss)
from src.models          import DockingModel
from src.utils           import random_rotation_matrices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def make_neg_pose(lig_pos, device, neg_shift_min, neg_shift_max):
    """Random rotation + large translation to make a non-binding negative pose."""
    B = lig_pos.shape[0]
    R   = random_rotation_matrices(B, device)
    com = lig_pos.mean(dim=1, keepdim=True)
    rot = torch.einsum("bij,bnj->bni", R, lig_pos - com) + com
    mag = torch.empty(B, 1, 1, device=device).uniform_(neg_shift_min, neg_shift_max)
    dirv = torch.randn(B, 1, 3, device=device)
    dirv = dirv / dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return rot + dirv * mag


def train_epoch(model, loader, optimizer, scaler, device, epoch, args):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        rec_feats     = batch["rec_feats"].to(device)
        rec_centers   = batch["rec_centers"].to(device)
        pocket_labels = batch["pocket_labels"].to(device)
        rec_mask      = batch["rec_mask"].to(device)
        lig_pos       = batch["lig_pos"].to(device)
        lig_atom_type = batch["lig_atom_type"].to(device)
        lig_mask      = batch["lig_mask"].to(device)
        affinity_gt   = batch["affinity"].to(device)
        bind_gt       = batch["bind_label"].to(device).squeeze(-1)

        # edge lists: move to device per-sample
        ei_list = [e.to(device) for e in batch["lig_edge_index"]]
        et_list = [e.to(device) for e in batch["lig_edge_type"]]
        B = rec_feats.shape[0]

        if args.use_negative_pose:
            lig_pos_neg = make_neg_pose(lig_pos, device,
                                        args.neg_shift_min, args.neg_shift_max)
            rec_feats_all   = torch.cat([rec_feats,    rec_feats],   0)
            rec_centers_all = torch.cat([rec_centers,  rec_centers], 0)
            rec_mask_all    = torch.cat([rec_mask,     rec_mask],    0)
            pocket_all      = torch.cat([pocket_labels,pocket_labels],0)
            lig_pos_all     = torch.cat([lig_pos,      lig_pos_neg], 0)
            lig_type_all    = torch.cat([lig_atom_type,lig_atom_type],0)
            lig_mask_all    = torch.cat([lig_mask,     lig_mask],    0)
            ei_all          = ei_list + ei_list
            et_all          = et_list + et_list
            bind_all        = torch.cat([bind_gt,
                                torch.zeros_like(bind_gt)], 0)
            aff_all         = torch.cat([affinity_gt.squeeze(-1),
                                torch.zeros_like(affinity_gt.squeeze(-1))], 0)
            is_pos          = torch.cat([
                torch.ones(B,  dtype=torch.bool, device=device),
                torch.zeros(B, dtype=torch.bool, device=device)], 0)
        else:
            rec_feats_all   = rec_feats
            rec_centers_all = rec_centers
            rec_mask_all    = rec_mask
            pocket_all      = pocket_labels
            lig_pos_all     = lig_pos
            lig_type_all    = lig_atom_type
            lig_mask_all    = lig_mask
            ei_all          = ei_list
            et_all          = et_list
            bind_all        = bind_gt
            aff_all         = affinity_gt.squeeze(-1)
            is_pos          = torch.ones(B, dtype=torch.bool, device=device)

        ctx = torch.amp.autocast("cuda") if args.amp else nullcontext()
        with ctx:
            out = model(
                rec_feats_all, rec_centers_all, rec_mask_all,
                lig_pos_all, lig_type_all,
                ei_all, et_all, lig_mask_all)

            l_pocket  = pocket_bce_loss(
                out["pocket_logits"], pocket_all, rec_mask_all)
            l_bind    = binding_bce_loss(out["bind_logit"], bind_all)
            l_aff     = affinity_loss(out["affinity_pred"], aff_all, is_pos)
            l_contact = contact_loss(
                out["rec_tokens"], out["lig_tokens"],
                out["rec_centers"], out["lig_pos"],
                out["rec_mask"],    out["lig_mask"],
                contact_thresh=args.contact_thresh)
            l_flex    = flex_loss(
                out["rec_tokens"], out["rec_centers"],
                out["rec_mask"],   k_nbr=args.flex_knn)

            loss = (args.w_pocket  * l_pocket  +
                    args.w_bind    * l_bind    +
                    args.w_aff     * l_aff     +
                    args.w_contact * l_contact +
                    args.w_flex    * l_flex)

        optimizer.zero_grad(set_to_none=True)
        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total += loss.item(); n += 1
        if n % 20 == 0:
            logger.info("Epoch %d step %d | total=%.4f pocket=%.4f "
                        "bind=%.4f aff=%.4f contact=%.4f flex=%.4f",
                        epoch, n, loss.item(), l_pocket.item(),
                        l_bind.item(), l_aff.item(),
                        l_contact.item(), l_flex.item())
    return total / max(n, 1)


def build_parser():
    p = argparse.ArgumentParser(description="Stage 3: Protein-Ligand Supervised Fine-tuning")
    p.add_argument("--index_file",       required=True)
    p.add_argument("--npz_root",         required=True)
    p.add_argument("--save_dir",         default="./outputs/stage3")
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--seq_len",          type=int,   default=512)
    p.add_argument("--K",                type=int,   default=32)
    p.add_argument("--d_model",          type=int,   default=256)
    p.add_argument("--nhead",            type=int,   default=8)
    p.add_argument("--nlayers",          type=int,   default=6)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-2)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--workers",          type=int,   default=4)
    p.add_argument("--w_pocket",         type=float, default=1.0)
    p.add_argument("--w_bind",           type=float, default=1.0)
    p.add_argument("--w_aff",            type=float, default=1.0)
    p.add_argument("--w_contact",        type=float, default=1.0)
    p.add_argument("--w_flex",           type=float, default=0.1)
    p.add_argument("--contact_thresh",   type=float, default=6.0)
    p.add_argument("--flex_knn",         type=int,   default=8)
    p.add_argument("--pocket_extra",     type=float, default=2.0)
    p.add_argument("--use_negative_pose",action="store_true")
    p.add_argument("--neg_shift_min",    type=float, default=20.0)
    p.add_argument("--neg_shift_max",    type=float, default=40.0)
    p.add_argument("--save_every",       type=int,   default=5)
    p.add_argument("--seed",             type=int,   default=2024)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp",              action="store_true")
    p.add_argument("--pretrained_stage2",default="")
    p.add_argument("--resume",           default="")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    ds_train = ProteinLigandDataset(
        args.index_file, args.npz_root,
        seq_len=args.seq_len, K=args.K, split="train", seed=args.seed)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, collate_fn=finetune_collate)

    model = DockingModel(
        d_model=args.d_model, nhead=args.nhead,
        nlayers=args.nlayers, dropout=args.dropout).to(device)

    if args.pretrained_stage2:
        load_stage2_encoder(model, args.pretrained_stage2,
                            encoder_attr_rec="rec_encoder",
                            encoder_attr_lig=None)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scaler)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(
            model, dl_train, optimizer, scaler, device, epoch, args)
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
    logger.info("Stage 3 training complete.")


if __name__ == "__main__":
    main()
