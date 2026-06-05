# -*- coding: utf-8 -*-
"""
Training utilities: seed, LR warmup, checkpoint save/load, AMP context.
Self-contained - no imports from other projects.
"""

import os
import math
import logging
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def save_checkpoint(state: dict, save_dir: str, tag: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ckpt_{tag}.pt")
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(path: str, model: nn.Module,
                    optimizer=None, scaler=None):
    """Load state dict into model (and optionally optimizer/scaler).
    Returns (start_epoch, global_step).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch  = ckpt.get("epoch", 0) + 1
    global_step  = ckpt.get("global_step", 0)
    logger.info("Resumed from %s at epoch %d, step %d", path, start_epoch, global_step)
    return start_epoch, global_step


def amp_context(enabled: bool):
    """Return autocast context (or nullcontext if disabled)."""
    if enabled:
        return torch.amp.autocast("cuda")
    return nullcontext()


def load_stage2_encoder(
    model,
    ckpt_path: str,
    encoder_attr_rec: str = "encoder.rec_encoder",
    encoder_attr_lig: Optional[str] = "encoder.lig_encoder",
) -> None:
    """
    Load the surface encoder weights from a Stage-2 SurfVQMAE checkpoint
    into the receptor (and optionally ligand) encoder of the Stage-3 model.

    The Stage-2 checkpoint stores keys like:
        local.*   -> PointMLP
        blocks.*  -> SurfFormerBlocks
    We copy these into the Stage-3 model's encoder sub-modules.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["model"]

    # Extract encoder-related keys (local + blocks, skip codebook/decoder)
    # Stage-2 SurfVQMAE stores encoder weights under "encoder.local.*" / "encoder.blocks.*"
    enc_keys = {}
    for k, v in sd.items():
        if k.startswith("encoder.local.") or k.startswith("encoder.blocks.") or k.startswith("encoder.norm"):
            enc_keys[k[len("encoder."):]] = v   # strip "encoder." prefix
        elif k.startswith("local.") or k.startswith("blocks.") or k.startswith("norm."):
            enc_keys[k] = v                      # already correct format

    def _copy(attr: str):
        module = model
        for part in attr.split("."):
            module = getattr(module, part)
        result = module.load_state_dict(enc_keys, strict=False)
        logger.info("Loaded Stage-2 encoder into %s: missing=%s, unexpected=%s",
                    attr, result.missing_keys, result.unexpected_keys)

    _copy(encoder_attr_rec)
    if encoder_attr_lig:
        try:
            _copy(encoder_attr_lig)
        except AttributeError:
            logger.warning("No ligand encoder attribute %s; skipping.", encoder_attr_lig)
