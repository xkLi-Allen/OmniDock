# -*- coding: utf-8 -*-
"""
Stage 2: Backbone-aware Surface VQ-MAE Pretraining.

Dual-branch architecture:
  - Surface branch: PointMLP + SurfFormer + VQ codebook + decoder (masked reconstruction)
  - Backbone branch: BackboneEncoder + TorsionHead + SSHead
  - Cross-alignment between surface tokens and backbone tokens

The model learns BOTH surface geometry AND protein backbone structure.
"""

import os
import sys
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_components import (
    SurfaceEncoder, BackboneEncoder, TorsionHead, SSHead,
    GumbelCodebook, PatchDecoder,
)
from stage2_dataset import BackboneAwareSurfaceDataset, stage2_collate_fn


class BackboneAwareSurfVQMAE(nn.Module):
    """
    Dual-branch VQ-MAE:
      - Surface branch: encoder + VQ codebook + decoder
      - Backbone branch: encoder + torsion/SS heads
      - Cross-alignment between surface and backbone tokens
    """

    def __init__(self, d_model=256, nhead=8, nlayers_surf=6, nlayers_bb=4,
                 K=50, num_codes=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.surf_encoder = SurfaceEncoder(
            in_dim=6, d_model=d_model, nhead=nhead, nlayers=nlayers_surf, dropout=dropout
        )
        self.pre_code = nn.LayerNorm(d_model)
        self.to_logits = nn.Linear(d_model, num_codes)
        self.codebook = GumbelCodebook(num_codes=num_codes, code_dim=d_model)
        self.decoder = PatchDecoder(d_model=d_model, K=K, hidden=2 * d_model)

        self.bb_encoder = BackboneEncoder(
            d_model=d_model, nhead=nhead, nlayers=nlayers_bb, dropout=dropout
        )
        self.torsion_head = TorsionHead(d_model=d_model, hidden=d_model)
        self.ss_head = SSHead(d_model=d_model, hidden=128)

        self.surf_proj = nn.Linear(d_model, d_model)
        self.bb_proj = nn.Linear(d_model, d_model)

    def forward(self, surface_feats, surface_centers, surface_mask, surface_pad,
                bb_feats, bb_valid, tau=1.0):
        surf_tokens = self.surf_encoder(surface_feats, surface_centers, key_padding_mask=surface_pad)
        surf_pre = self.pre_code(surf_tokens)

        logits = self.to_logits(surf_pre)
        zq, post = self.codebook(logits, tau=tau, hard=True)

        maskf = surface_mask.float().unsqueeze(-1)
        tokens_mixed = surf_pre * (1 - maskf) + zq * maskf
        surf_recon = self.decoder(tokens_mixed)

        bb_pad_mask = ~bb_valid
        bb_tokens = self.bb_encoder(bb_feats, mask=bb_pad_mask)
        torsion_pred = self.torsion_head(bb_tokens)
        ss_pred = self.ss_head(bb_tokens)

        surf_align = self.surf_proj(surf_tokens)
        bb_align = self.bb_proj(bb_tokens)

        return {
            'surf_recon': surf_recon,
            'surf_logits': logits,
            'surf_post': post,
            'surf_tokens': surf_tokens,
            'torsion_pred': torsion_pred,
            'ss_pred': ss_pred,
            'bb_tokens': bb_tokens,
            'surf_align': surf_align,
            'bb_align': bb_align,
        }


def chamfer_distance(a, b):
    """a, b: (N, K, 3). Returns mean Chamfer (squared)."""
    with torch.amp.autocast('cuda', enabled=False):
        a32, b32 = a.float(), b.float()
        D = torch.cdist(a32, b32)
        a2b = D.min(dim=2).values
        b2a = D.min(dim=1).values
        return (a2b.pow(2).mean(dim=1) + b2a.pow(2).mean(dim=1)).mean()


def compute_stage2_losses(model_out, batch, tau, w_chamfer, w_kl, w_torsion, w_ss, w_align):
    device = model_out['surf_recon'].device

    surf_recon = model_out['surf_recon']
    B, T, K, _ = surf_recon.shape
    smask = batch['surface_mask'].to(device)
    m = smask.reshape(B * T)

    if m.any():
        rec_m = surf_recon.reshape(B * T, K, 3)[m]
        tgt_m = batch['surface_coords'].to(device).reshape(B * T, K, 3)[m]
        l_chamfer = chamfer_distance(rec_m, tgt_m)
    else:
        l_chamfer = torch.tensor(0.0, device=device)

    logits = model_out['surf_logits']
    C = logits.shape[-1]
    q = torch.softmax(logits / max(1e-4, tau), dim=-1)
    q_flat = q.reshape(B * T, C)[m] if m.any() else q.reshape(B * T, C)[:1]
    logC = math.log(C)
    l_kl = (q_flat.clamp_min(1e-9).log().mul(q_flat).sum(dim=-1) + logC).mean()

    torsion_pred = model_out['torsion_pred']
    torsion_target = batch['bb_torsion_target'].to(device)
    torsion_valid = batch['torsion_valid'].to(device)
    if torsion_valid.any():
        l_torsion = F.mse_loss(torsion_pred[torsion_valid], torsion_target[torsion_valid])
    else:
        l_torsion = torch.tensor(0.0, device=device)

    ss_pred = model_out['ss_pred']
    ss_target = batch['bb_ss_target'].to(device)
    bb_valid = batch['bb_valid'].to(device)
    if bb_valid.any():
        l_ss = F.cross_entropy(ss_pred[bb_valid], ss_target[bb_valid])
    else:
        l_ss = torch.tensor(0.0, device=device)

    surf_align = model_out['surf_align']
    bb_align = model_out['bb_align']
    surf2res = batch['surf2res'].to(device)
    spad = batch['surface_pad'].to(device)

    l_align = torch.tensor(0.0, device=device)
    n_align = 0
    for b in range(B):
        valid_surf = ~spad[b]
        valid_bb = batch['bb_valid'][b].to(device)
        if valid_surf.sum() == 0 or valid_bb.sum() == 0:
            continue
        s_emb = F.normalize(surf_align[b][valid_surf], dim=-1)
        b_emb = F.normalize(bb_align[b][valid_bb], dim=-1)
        s2r_b = surf2res[b][valid_surf].clamp(max=b_emb.shape[0] - 1)
        target_emb = b_emb[s2r_b]
        sim = (s_emb * target_emb).sum(dim=-1)
        l_align = l_align + (1.0 - sim).mean()
        n_align += 1

    if n_align > 0:
        l_align = l_align / n_align

    total = (w_chamfer * l_chamfer + w_kl * l_kl + w_torsion * l_torsion +
             w_ss * l_ss + w_align * l_align)

    return {
        'total': total,
        'chamfer': l_chamfer.item(),
        'kl': l_kl.item(),
        'torsion': l_torsion.item(),
        'ss': l_ss.item(),
        'align': l_align.item(),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description="Stage 2: Backbone-aware Surface VQ-MAE Pretraining")
    ap.add_argument("--data_root", type=str,
                    default="/data/jiangjiaqi/srzhang/InversionDock/Data/Processed_skempi_backbone_aware")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_residues", type=int, default=256)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers_surf", type=int, default=6)
    ap.add_argument("--nlayers_bb", type=int, default=4)
    ap.add_argument("--num_codes", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=5000)
    ap.add_argument("--tau_init", type=float, default=1.0)
    ap.add_argument("--tau_min", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--w_chamfer", type=float, default=1.0)
    ap.add_argument("--w_kl", type=float, default=1e-3)
    ap.add_argument("--w_torsion", type=float, default=5.0)
    ap.add_argument("--w_ss", type=float, default=2.0)
    ap.add_argument("--w_align", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, default="./ckpts_stage2_backbone_aware")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    ds = BackboneAwareSurfaceDataset(
        args.data_root, seq_len=args.seq_len, K=args.K,
        mask_ratio=args.mask_ratio, max_residues=args.max_residues,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    drop_last=True, collate_fn=stage2_collate_fn)

    model = BackboneAwareSurfVQMAE(
        d_model=args.d_model, nhead=args.nhead,
        nlayers_surf=args.nlayers_surf, nlayers_bb=args.nlayers_bb,
        K=args.K, num_codes=args.num_codes, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        avg_loss = 0.0
        p = epoch / max(1, args.epochs - 1)
        tau = args.tau_init * (1 - p) + args.tau_min * p

        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for it, batch in enumerate(pbar):
            lr = min(args.lr, args.lr * (global_step + 1) / max(1, args.warmup_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            with torch.amp.autocast('cuda', enabled=args.amp):
                out = model(
                    surface_feats=batch['surface_feats'].to(device),
                    surface_centers=batch['surface_centers'].to(device),
                    surface_mask=batch['surface_mask'].to(device),
                    surface_pad=batch['surface_pad'].to(device),
                    bb_feats=batch['bb_feats'].to(device),
                    bb_valid=batch['bb_valid'].to(device),
                    tau=tau,
                )
                losses = compute_stage2_losses(
                    out, batch, tau,
                    args.w_chamfer, args.w_kl, args.w_torsion, args.w_ss, args.w_align
                )
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

            avg_loss = 0.9 * avg_loss + 0.1 * loss.item() if it > 0 else loss.item()
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}", 'cd': f"{losses['chamfer']:.4f}",
                'tor': f"{losses['torsion']:.4f}", 'ss': f"{losses['ss']:.3f}",
            })
            global_step += 1

        print(f"[Epoch {epoch}] avg_loss = {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"e{epoch:03d}.pt")
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optim": optimizer.state_dict(), "global_step": global_step,
            }, path)
            print(f"[Save] {path}")

    final = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "epoch": args.epochs - 1, "model": model.state_dict(),
        "optim": optimizer.state_dict(), "global_step": global_step,
    }, final)
    print(f"[Done] {final}")


if __name__ == "__main__":
    main()
