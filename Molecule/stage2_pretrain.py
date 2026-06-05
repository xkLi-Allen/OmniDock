import os
import sys
import math
import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.train_utils import set_seed, save_checkpoint, load_checkpoint, warmup_lr
from src.datasets    import SurfacePretrainDataset, pretrain_collate
from src.losses      import chamfer_distance, curvature_proxy, kl_to_uniform
from src.models      import SurfVQMAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, scaler, device, epoch, args, global_step):
    model.train()
    total_loss = 0.0
    n = 0
    for feats, coords, centers, mask in loader:
        feats   = feats.to(device,   non_blocking=True)
        coords  = coords.to(device,  non_blocking=True)
        centers = centers.to(device, non_blocking=True)
        mask    = mask.to(device,    non_blocking=True)

        lr = warmup_lr(global_step, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        ctx = torch.amp.autocast("cuda") if args.amp else nullcontext()
        with ctx:
            rec, curv_pred, logits, post = model(
                feats, centers, mask, tau=args.tau, hard=True)
            B, T, K, _ = rec.shape
            m = mask.view(B * T)
            if m.any():
                rec_m  = rec.view(B * T, K, 3)[m]
                tgt_m  = coords.view(B * T, K, 3)[m]
                l_cd   = chamfer_distance(rec_m, tgt_m)
                curv_t = curvature_proxy(tgt_m).detach()
                curv_p = curv_pred.view(B * T)[m]
                l_curv = F.mse_loss(curv_p, curv_t)
                l_kl   = kl_to_uniform(logits.view(B * T, -1)[m], args.tau)
            else:
                l_cd = l_curv = l_kl = rec.new_tensor(0.0)
            loss = l_cd + args.curv_weight * l_curv + args.kl_weight * l_kl

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
        n += 1
        global_step += 1

        if n % 50 == 0:
            logger.info("Epoch %d | step %d | lr=%.2e | loss=%.4f cd=%.4f kl=%.4f curv=%.4f",
                        epoch, global_step, lr, loss.item(),
                        l_cd.item(), l_kl.item(), l_curv.item())
    return total_loss / max(n, 1), global_step


def build_parser():
    p = argparse.ArgumentParser(description="Stage 2: Surface VQ-MAE Pretraining")
    p.add_argument("--data_root",     required=True)
    p.add_argument("--save_dir",      default="./outputs/stage2")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--seq_len",       type=int,   default=512)
    p.add_argument("--K",             type=int,   default=32)
    p.add_argument("--mask_ratio",    type=float, default=0.60)
    p.add_argument("--d_model",       type=int,   default=256)
    p.add_argument("--nhead",         type=int,   default=8)
    p.add_argument("--nlayers",       type=int,   default=6)
    p.add_argument("--codebook_size", type=int,   default=2048)
    p.add_argument("--codebook_dim",  type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--weight_decay",  type=float, default=0.0)
    p.add_argument("--warmup_steps",  type=int,   default=1000)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--tau_min",       type=float, default=0.5)
    p.add_argument("--kl_weight",     type=float, default=1e-3)
    p.add_argument("--curv_weight",   type=float, default=0.1)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--save_every",    type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=2024)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--resume",        default="")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    ds = SurfacePretrainDataset(
        args.data_root, seq_len=args.seq_len, K=args.K,
        mask_ratio=args.mask_ratio, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    drop_last=True, collate_fn=pretrain_collate)

    model = SurfVQMAE(
        in_dim=6, d_model=args.d_model, nhead=args.nhead,
        nlayers=args.nlayers, K=args.K,
        num_codes=args.codebook_size, code_dim=args.codebook_dim,
        dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scaler)

    tau_start = args.tau
    for epoch in range(start_epoch, args.epochs):
        # anneal tau
        p = epoch / max(args.epochs - 1, 1)
        args.tau = tau_start * (1 - p) + args.tau_min * p

        avg_loss, global_step = train_epoch(
            model, dl, optimizer, scaler, device, epoch, args, global_step)
        scheduler.step(avg_loss)
        logger.info("Epoch %d done | avg_loss=%.4f | tau=%.4f",
                    epoch, avg_loss, args.tau)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                "epoch": epoch, "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "args": vars(args),
            }, args.save_dir, f"e{epoch:04d}")

    save_checkpoint({
        "epoch": args.epochs - 1, "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if args.amp else None,
        "args": vars(args),
    }, args.save_dir, "final")
    logger.info("Stage 2 training complete.")


if __name__ == "__main__":
    main()
