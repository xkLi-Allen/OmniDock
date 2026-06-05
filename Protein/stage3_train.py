# -*- coding: utf-8 -*-
"""
Stage 3 Training: Pocket-conditioned Docking with Structure Generation.

Multi-task learning:
  - Pocket center/radius regression
  - Binding classification (with negative pose augmentation)
  - Affinity (ddG) regression
  - Surface complementarity
  - Ligand torsion denoising (pocket-conditioned structure generation)
  - Ligand secondary structure prediction
"""

import os
import sys
import math
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage3_dataset import SkempiDockingDataset, stage3_collate_fn
from stage3_model import (
    DockingModel, random_rotation_matrices,
    surface_complementarity_loss, load_pretrained_encoders,
)


def compute_stage3_losses(out, batch, is_pos, args):
    device = out['bind_logit'].device

    pc_pred = out['pocket_center']
    pr_pred = out['pocket_radius']
    pocket_center_gt = batch['pocket_center'].to(device)
    pocket_radius_gt = batch['pocket_radius'].to(device)

    l_pocket = F.smooth_l1_loss(pc_pred, pocket_center_gt) + F.smooth_l1_loss(pr_pred, pocket_radius_gt)

    bind_label = batch['bind_label'].to(device)
    l_bind = F.binary_cross_entropy_with_logits(out['bind_logit'], bind_label)

    affinity_gt = batch['affinity'].to(device)
    if is_pos.any():
        l_aff = F.smooth_l1_loss(out['aff_pred'][is_pos], affinity_gt[is_pos])
    else:
        l_aff = torch.tensor(0.0, device=device)

    l_comp = surface_complementarity_loss(
        out['rec_surf_tokens'], out['lig_surf_tokens'],
        out['rec_centers'], out['lig_centers'],
        out['rec_pad'], out['lig_pad'],
        pc_pred, pr_pred,
        contact_thresh=args.contact_thresh,
    )

    lig_bb_mask = batch['lig_bb_mask'].to(device)
    torsion_target = batch['lig_torsion_target'].to(device)
    if is_pos.any() and lig_bb_mask[is_pos].any():
        valid = lig_bb_mask[is_pos]
        l_torsion = F.mse_loss(out['torsion_pred'][is_pos][valid], torsion_target[is_pos][valid])
    else:
        l_torsion = torch.tensor(0.0, device=device)

    ss_target = batch['lig_ss_target'].to(device)
    if is_pos.any() and lig_bb_mask[is_pos].any():
        valid = lig_bb_mask[is_pos]
        l_ss = F.cross_entropy(out['ss_pred'][is_pos][valid], ss_target[is_pos][valid])
    else:
        l_ss = torch.tensor(0.0, device=device)

    total = (args.w_pocket * l_pocket + args.w_bind * l_bind + args.w_aff * l_aff +
             args.w_comp * l_comp + args.w_torsion * l_torsion + args.w_ss * l_ss)

    return {
        'total': total,
        'pocket': l_pocket.item(), 'bind': l_bind.item(), 'aff': l_aff.item(),
        'comp': l_comp.item() if isinstance(l_comp, torch.Tensor) else l_comp,
        'torsion': l_torsion.item(), 'ss': l_ss.item(),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description="Stage 3: Pocket-conditioned Docking Training")
    ap.add_argument("--skempi_csv", type=str,
                    default="/data/jiangjiaqi/srzhang/InversionDock/Data/Skempi_dataset/skempi_v2.csv")
    ap.add_argument("--npz_root", type=str,
                    default="/data/jiangjiaqi/srzhang/InversionDock/Data/Processed_skempi_backbone_aware")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_residues", type=int, default=256)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers_surf", type=int, default=6)
    ap.add_argument("--nlayers_bb", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--contact_thresh", type=float, default=5.0)
    ap.add_argument("--w_pocket", type=float, default=1.0)
    ap.add_argument("--w_bind", type=float, default=10.0)
    ap.add_argument("--w_aff", type=float, default=5.0)
    ap.add_argument("--w_comp", type=float, default=3.0)
    ap.add_argument("--w_torsion", type=float, default=8.0)
    ap.add_argument("--w_ss", type=float, default=3.0)
    ap.add_argument("--use_negative_pose", action="store_true")
    ap.add_argument("--neg_shift_min", type=float, default=15.0)
    ap.add_argument("--neg_shift_max", type=float, default=35.0)
    ap.add_argument("--save_dir", type=str, default="./ckpts_stage3_structure_aware")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained", type=str, default="", help="Stage 2 checkpoint path")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = SkempiDockingDataset(
        args.skempi_csv, args.npz_root, K=args.K,
        seq_len=args.seq_len, max_residues=args.max_residues,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=True, collate_fn=stage3_collate_fn)

    model = DockingModel(
        d_model=args.d_model, nhead=args.nhead,
        nlayers_surf=args.nlayers_surf, nlayers_bb=args.nlayers_bb,
        K=args.K, dropout=args.dropout,
    ).to(device)

    if args.pretrained:
        load_pretrained_encoders(model, args.pretrained, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for it, batch in enumerate(pbar):
            B = batch['rec_feats'].size(0)

            rec_feats = batch['rec_feats'].to(device)
            rec_centers = batch['rec_centers'].to(device)
            rec_pad = batch['rec_pad'].to(device)
            lig_feats = batch['lig_feats'].to(device)
            lig_centers = batch['lig_centers'].to(device)
            lig_pad = batch['lig_pad'].to(device)
            rec_bb_feats = batch['rec_bb_feats'].to(device)
            rec_bb_mask = batch['rec_bb_mask'].to(device)
            lig_bb_feats = batch['lig_bb_feats'].to(device)
            lig_bb_mask = batch['lig_bb_mask'].to(device)

            if args.use_negative_pose:
                R = random_rotation_matrices(B, device)
                com = lig_centers.mean(dim=1, keepdim=True)
                centered = lig_centers - com
                rotated = torch.einsum('bij,btj->bti', R, centered) + com
                direction = torch.randn(B, 3, device=device)
                direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                mag = torch.empty(B, 1, 1, device=device).uniform_(args.neg_shift_min, args.neg_shift_max)
                lig_centers_neg = rotated + direction.unsqueeze(1) * mag

                rel = lig_feats[..., :3]
                nrm = lig_feats[..., 3:]
                lig_feats_neg = torch.cat([
                    torch.einsum('bij,btkj->btki', R, rel),
                    torch.einsum('bij,btkj->btki', R, nrm),
                ], dim=-1)

                rec_feats_all = torch.cat([rec_feats, rec_feats], dim=0)
                rec_centers_all = torch.cat([rec_centers, rec_centers], dim=0)
                rec_pad_all = torch.cat([rec_pad, rec_pad], dim=0)
                lig_feats_all = torch.cat([lig_feats, lig_feats_neg], dim=0)
                lig_centers_all = torch.cat([lig_centers, lig_centers_neg], dim=0)
                lig_pad_all = torch.cat([lig_pad, lig_pad], dim=0)
                rec_bb_feats_all = torch.cat([rec_bb_feats, rec_bb_feats], dim=0)
                rec_bb_mask_all = torch.cat([rec_bb_mask, rec_bb_mask], dim=0)
                lig_bb_feats_all = torch.cat([lig_bb_feats, lig_bb_feats], dim=0)
                lig_bb_mask_all = torch.cat([lig_bb_mask, lig_bb_mask], dim=0)

                batch_aug = dict(batch)
                batch_aug['pocket_center'] = torch.cat([batch['pocket_center'], batch['pocket_center']], dim=0)
                batch_aug['pocket_radius'] = torch.cat([batch['pocket_radius'], batch['pocket_radius']], dim=0)
                batch_aug['bind_label'] = torch.cat([batch['bind_label'], torch.zeros(B)], dim=0)
                batch_aug['affinity'] = torch.cat([batch['affinity'], torch.zeros(B)], dim=0)
                batch_aug['lig_bb_mask'] = torch.cat([batch['lig_bb_mask'], batch['lig_bb_mask'][:B]], dim=0)
                batch_aug['lig_torsion_target'] = torch.cat([batch['lig_torsion_target'], batch['lig_torsion_target'][:B]], dim=0)
                batch_aug['lig_ss_target'] = torch.cat([batch['lig_ss_target'], batch['lig_ss_target'][:B]], dim=0)

                is_pos = torch.cat([torch.ones(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)], dim=0)
            else:
                rec_feats_all = rec_feats
                rec_centers_all = rec_centers
                rec_pad_all = rec_pad
                lig_feats_all = lig_feats
                lig_centers_all = lig_centers
                lig_pad_all = lig_pad
                rec_bb_feats_all = rec_bb_feats
                rec_bb_mask_all = rec_bb_mask
                lig_bb_feats_all = lig_bb_feats
                lig_bb_mask_all = lig_bb_mask
                batch_aug = batch
                is_pos = torch.ones(B, dtype=torch.bool)

            with torch.amp.autocast('cuda', enabled=args.amp):
                out = model(rec_feats_all, rec_centers_all, rec_pad_all,
                            lig_feats_all, lig_centers_all, lig_pad_all,
                            rec_bb_feats_all, rec_bb_mask_all,
                            lig_bb_feats_all, lig_bb_mask_all)
                losses = compute_stage3_losses(out, batch_aug, is_pos, args)
                loss = losses['total']

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

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{total_loss / (it + 1):.4f}",
                'bind': f"{losses['bind']:.3f}",
                'tor': f"{losses['torsion']:.3f}",
            })

        avg = total_loss / max(1, it + 1)
        print(f"[Epoch {epoch}] avg_loss = {avg:.4f}")

        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"e{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict()}, path)
            print(f"[Save] {path}")

    final = os.path.join(args.save_dir, "final.pt")
    torch.save({"epoch": args.epochs - 1, "model": model.state_dict(), "optim": optimizer.state_dict()}, final)
    print(f"[Done] {final}")


if __name__ == "__main__":
    main()
