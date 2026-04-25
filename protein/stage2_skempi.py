# -*- coding: utf-8 -*-
"""
VQ-MAE 无监督预训练（SKEMPI per-chain 版本）

依赖：
  - 同目录下已有的 unsupervised_pre_training.py

数据要求（单个 .npz）与原脚本一致：
  xs (M,3), ns (M,3),
  patch_centers (Nc,3), patch_knn_idx (Nc,K), patch_order (Nc,)
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

# 直接复用你之前写好的实现
from unsupervised_pre_training import (
    SurfacePatchDataset,   # 数据集类
    SurfVQMAE,             # VQ-MAE 模型
    set_seed,              # 随机种子
    collate_fn,            # batch pad
    train_one_epoch,       # 单 epoch 训练循环
    save_ckpt,             # 保存 ckpt
)

def main():
    ap = argparse.ArgumentParser(
        description="Surface-VQMAE pretraining on SKEMPI per-chain surfaces"
    )

    # 关键：默认 data_root 指向你新的 per-chain 目录
    ap.add_argument(
        "--data_root",
        type=str,
        default="/home/ai/zkchen/PytorchProjects/MagicPPI/Code-v3/Protein/Processed_pdbbind_per_chain",
        help="目录下应全部是 *.npz 的表面文件（Stage1 输出）",
    )

    # 跟原 unsupervised_pre_training.py 保持一致的一组超参
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--codebook_size", type=int, default=2048)
    ap.add_argument("--codebook_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau_min", type=float, default=0.5)
    ap.add_argument("--kl_weight", type=float, default=1e-3)
    ap.add_argument("--curv_weight", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)

    # 这里改一个默认保存目录，避免覆盖你之前的 ckpt
    ap.add_argument(
        "--save_dir",
        type=str,
        default="./ckpts_vqmae_pdbbind_per_chain",
    )
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default="")  # 继续训用

    args = ap.parse_args()

    # ----------------- 准备数据 & 模型 -----------------
    set_seed(args.seed)
    device = torch.device(args.device)

    ds = SurfacePatchDataset(
        args.data_root,
        seq_len=args.seq_len,
        K=args.K,
        mask_ratio=args.mask_ratio,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = SurfVQMAE(
        in_dim=6,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        K=args.K,
        num_codes=args.codebook_size,
        code_dim=args.codebook_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch = 0
    global_step = 0

    # 可选：从已有 ckpt 继续训
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and args.amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] Loaded from {args.resume} at epoch {start_epoch}")

    # ----------------- 训练循环 -----------------
    for epoch in range(start_epoch, args.epochs):
        # 和原脚本一样的 τ 退火策略
        p = epoch / max(1, args.epochs - 1)
        args.tau = args.tau * (1 - p) + args.tau_min * p

        loss, global_step = train_one_epoch(
            model,
            dl,
            optimizer,
            scaler,
            device,
            epoch,
            args,
            scheduler,
            global_step,
        )

        if (epoch + 1) % args.save_every == 0:
            path = save_ckpt(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "global_step": global_step,
                },
                args.save_dir,
                f"e{epoch:03d}",
            )
            print(f"[Save] {path}")

    # 最终 ckpt
    path = save_ckpt(
        {
            "epoch": args.epochs - 1,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
            "global_step": global_step,
        },
        args.save_dir,
        "final",
    )
    print(f"[Done] saved to {path}")


if __name__ == "__main__":
    main()
