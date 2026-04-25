
import os
import math
import json
import time
import random
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_warmup(step: int, warmup: int, base_lr: float) -> float:
    if warmup <= 0:
        return base_lr
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    return base_lr


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Uniform-ish random rotation matrix via random quaternion."""
    u1 = torch.rand((), device=device)
    u2 = torch.rand((), device=device)
    u3 = torch.rand((), device=device)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    # quaternion (x,y,z,w)
    x, y, z, w = q1, q2, q3, q4
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)]),
        torch.stack([2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)]),
        torch.stack([2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]),
    ], dim=0)
    return R


# -----------------------------
# Data
# -----------------------------

class SurfacePatchDataset(Dataset):
    """Streams .npz surfaces and yields one window along Morton order per item.

    Returns:
      feats:  (T,K,6) float32  -> [rel_xyz(3), normals(3)]
      coords: (T,K,3) float32  -> reconstruction target (relative xyz)
      centers:(T,3)  float32   -> patch centers (for attention bias)
      mask:   (T,)   bool      -> masked patches
    """

    def __init__(
        self,
        root: str,
        seq_len: int = 512,
        K: int = 50,
        mask_ratio: float = 0.6,
        cache_meta: bool = True,
        random_rotate: bool = False,
    ):
        super().__init__()
        self.root = root
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".npz")
        ])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz found under {root}")

        self.seq_len = int(seq_len)
        self.K = int(K)
        self.mask_ratio = float(mask_ratio)
        self.cache_meta = bool(cache_meta)
        self.random_rotate = bool(random_rotate)
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def _load_file(self, path: str):
        if self.cache_meta and path in self._cache:
            return self._cache[path]
        with np.load(path, allow_pickle=True) as data:
            xs = data["xs"].astype(np.float32)  # (M,3)
            ns = data["ns"].astype(np.float32)  # (M,3)
            centers = data["patch_centers"].astype(np.float32)  # (Nc,3)
            knn = data["patch_knn_idx"].astype(np.int64)  # (Nc,K0)
            # prefer patch_order
            if "patch_order" in data:
                order = data["patch_order"].astype(np.int64)
            elif "patch_morton" in data:
                order = np.argsort(data["patch_morton"].astype(np.uint64)).astype(np.int64)
            else:
                raise KeyError(f"{path} has no patch_order/patch_morton")
            meta = json.loads(str(data["meta"])) if "meta" in data else {}

        # force K
        K0 = knn.shape[1]
        if K0 < self.K:
            pad = np.tile(knn[:, -1:], (1, self.K - K0))
            knn = np.concatenate([knn, pad], axis=1)
        elif K0 > self.K:
            knn = knn[:, : self.K]

        out = dict(xs=xs, ns=ns, centers=centers, knn=knn, order=order, meta=meta)
        if self.cache_meta:
            self._cache[path] = out
        return out

    def __getitem__(self, idx: int):
        path = self.files[idx % len(self.files)]
        arr = self._load_file(path)
        xs, ns, centers, knn, order = arr["xs"], arr["ns"], arr["centers"], arr["knn"], arr["order"]

        Nc = centers.shape[0]
        if Nc <= 0:
            raise ValueError(f"No patch centers in {path}")

        # choose a contiguous window along Morton order
        if Nc <= self.seq_len:
            sel = order
        else:
            start = np.random.randint(0, Nc - self.seq_len + 1)
            sel = order[start : start + self.seq_len]

        pts_idx = knn[sel]  # (T,K)
        ctrs = centers[sel]  # (T,3)

        rel_xyz = xs[pts_idx] - ctrs[:, None, :]  # (T,K,3)
        norms = ns[pts_idx]  # (T,K,3)

        feats = np.concatenate([rel_xyz, norms], axis=-1).astype(np.float32)  # (T,K,6)
        coords = rel_xyz.astype(np.float32)  # (T,K,3)

        T = feats.shape[0]
        mask = np.zeros((T,), dtype=np.bool_)
        num_mask = max(1, int(round(self.mask_ratio * T)))
        masked_idx = np.random.choice(T, size=num_mask, replace=False)
        mask[masked_idx] = True

        # optional SO(3) augmentation (rotate rel_xyz + normals together)
        if self.random_rotate:
            # use torch here to avoid numerical drift
            R = random_rotation_matrix(device=torch.device("cpu")).numpy().astype(np.float32)  # (3,3)
            rel_xyz_r = rel_xyz @ R.T
            norms_r = norms @ R.T
            feats = np.concatenate([rel_xyz_r, norms_r], axis=-1).astype(np.float32)
            coords = rel_xyz_r.astype(np.float32)

        return {
            "feats": torch.from_numpy(feats),
            "coords": torch.from_numpy(coords),
            "centers": torch.from_numpy(ctrs),
            "mask": torch.from_numpy(mask),
            "name": os.path.basename(path),
        }


def collate_fn(batch):
    maxT = max(b["feats"].shape[0] for b in batch)
    K = batch[0]["feats"].shape[1]
    Xs, Ys, Cs, Ms = [], [], [], []
    for b in batch:
        T = b["feats"].shape[0]
        padT = maxT - T
        if padT > 0:
            pad_x = torch.zeros((padT, K, 6), dtype=b["feats"].dtype)
            pad_y = torch.zeros((padT, K, 3), dtype=b["coords"].dtype)
            pad_c = torch.zeros((padT, 3), dtype=b["centers"].dtype)
            pad_m = torch.zeros((padT,), dtype=b["mask"].dtype)
            Xs.append(torch.cat([b["feats"], pad_x], dim=0))
            Ys.append(torch.cat([b["coords"], pad_y], dim=0))
            Cs.append(torch.cat([b["centers"], pad_c], dim=0))
            Ms.append(torch.cat([b["mask"], pad_m], dim=0))
        else:
            Xs.append(b["feats"])
            Ys.append(b["coords"])
            Cs.append(b["centers"])
            Ms.append(b["mask"])

    Xs = torch.stack(Xs, dim=0)  # (B,T,K,6)
    Ys = torch.stack(Ys, dim=0)  # (B,T,K,3)
    Cs = torch.stack(Cs, dim=0)  # (B,T,3)
    Ms = torch.stack(Ms, dim=0)  # (B,T)
    return Xs, Ys, Cs, Ms


# -----------------------------
# Geometry helpers
# -----------------------------

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a,b: (N,K,3). Returns mean Chamfer (squared)."""
    # force fp32 for cdist stability
    with torch.cuda.amp.autocast(enabled=False):
        a32 = a.float()
        b32 = b.float()
        D = torch.cdist(a32, b32)  # (N,K,K)
        a2b = D.min(dim=2).values
        b2a = D.min(dim=1).values
        cd = (a2b.pow(2).mean(dim=1) + b2a.pow(2).mean(dim=1)).mean()
    return cd


def curvature_proxy(coords: torch.Tensor) -> torch.Tensor:
    """coords: (N,K,3) -> (N,) curvature proxy (lambda_min/sum(lambda))."""
    with torch.cuda.amp.autocast(enabled=False):
        x = coords.float()
        N, K, _ = x.shape
        x = x - x.mean(dim=1, keepdim=True)
        denom = float(K - 1) if K > 1 else 1.0
        cov = torch.einsum("nki,nkj->nij", x, x) / (denom + 1e-6)  # (N,3,3)
        evals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        kappa = evals[:, 0] / (evals.sum(dim=1) + 1e-8)
    return kappa.to(coords.dtype)


# -----------------------------
# Models
# -----------------------------

class PointMLP(nn.Module):
    def __init__(self, in_dim=6, hidden=128, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):  # (B,T,K,6)
        B, T, K, _ = x.shape
        x = x.reshape(B * T * K, -1)
        h = self.mlp(x).reshape(B, T, K, -1)
        h = h.max(dim=2).values  # (B,T,D)
        return self.norm(h)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.shape[1]
        return self.pe[:T].unsqueeze(0).to(x.dtype)


class RBFGeodesicBias(nn.Module):
    """RBF embedding of pairwise Euclidean distances between patch centers -> per-head bias."""

    def __init__(self, nhead, num_rbf=16):
        super().__init__()
        centers = torch.linspace(0.0, 60.0, num_rbf)  # Å
        widths = torch.ones_like(centers) * (centers[1] - centers[0] + 1e-6)
        self.register_buffer("mu", centers)
        self.register_buffer("beta", 1.0 / (2 * widths**2))
        self.proj = nn.Linear(num_rbf, nhead)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:  # (B,T,3)
        with torch.no_grad():
            D = torch.cdist(centers, centers)  # (B,T,T)
        diff = D.unsqueeze(-1) - self.mu.view(1, 1, 1, -1)
        rbf = torch.exp(-self.beta.view(1, 1, 1, -1) * diff.pow(2))
        bias = self.proj(rbf).permute(0, 3, 1, 2)  # (B,heads,T,T)
        return bias


class SurfFormerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_ff=1024, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.drop_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        self.bias = RBFGeodesicBias(nhead=nhead)
        self.pos = SinusoidalPositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, centers: torch.Tensor, key_padding_mask=None):
        B, T, D = x.shape
        x = x + self.pos(x)

        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_bias = self.bias(centers).to(x.dtype)

        # try SDPA first
        try:
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.drop_p if self.training else 0.0,
                is_causal=False,
            )
        except TypeError:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + attn_bias
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v

        attn_out = attn_out.transpose(2, 1).contiguous().view(B, T, D)
        x = x + self.resid_drop(self.proj_out(attn_out))
        x = x + self.ff(self.norm2(x))
        return x


class GumbelCodebook(nn.Module):
    def __init__(self, num_codes=2048, code_dim=256):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim) * 0.02)
        self.num_codes = num_codes
        self.code_dim = code_dim

    def forward(self, logits: torch.Tensor, tau=1.0, hard=True):
        g = -torch.empty_like(logits).exponential_().log()  # gumbel
        y = F.softmax((logits + g) / max(1e-4, tau), dim=-1)
        if hard:
            idx = y.argmax(dim=-1)
            y_hard = F.one_hot(idx, num_classes=logits.shape[-1]).type_as(y)
            y = (y_hard - y).detach() + y
        z = y @ self.codebook
        return z, y


class PatchDecoder(nn.Module):
    def __init__(self, d_model=256, K=50, hidden=512):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, K * 3)
        )

    def forward(self, tokens: torch.Tensor):
        out = self.mlp(tokens)
        return out.view(tokens.shape[0], tokens.shape[1], self.K, 3)


class CurvatureHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor):
        return self.net(tokens).squeeze(-1)


class SurfVQMAE(nn.Module):
    def __init__(
        self,
        in_dim=6,
        d_model=256,
        nhead=8,
        nlayers=6,
        K=50,
        num_codes=2048,
        code_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.local = PointMLP(in_dim=in_dim, hidden=d_model, out_dim=d_model)
        self.blocks = nn.ModuleList([
            SurfFormerBlock(d_model, nhead, 4 * d_model, dropout) for _ in range(nlayers)
        ])
        self.pre_code = nn.LayerNorm(d_model)
        self.to_logits = nn.Linear(d_model, num_codes)
        self.codebook = GumbelCodebook(num_codes=num_codes, code_dim=code_dim)
        self.up = nn.Linear(code_dim, d_model) if code_dim != d_model else nn.Identity()
        self.decoder = PatchDecoder(d_model=d_model, K=K, hidden=2 * d_model)
        self.curv = CurvatureHead(d_model)

    def forward(self, feats: torch.Tensor, centers: torch.Tensor, mask: torch.Tensor, tau=1.0, hard=True):
        x = self.local(feats)  # (B,T,D)
        for blk in self.blocks:
            x = blk(x, centers)
        x = self.pre_code(x)

        logits = self.to_logits(x)  # (B,T,C)
        zq, post = self.codebook(logits, tau=tau, hard=hard)  # (B,T,Dc)
        zq = self.up(zq)

        maskf = mask.float().unsqueeze(-1)
        tokens = x * (1 - maskf) + zq * maskf

        rec = self.decoder(tokens)
        curv = self.curv(tokens)
        return rec, curv, logits, post


# -----------------------------
# Train
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args,
    scheduler=None,
    global_step: int = 0,
):
    model.train()
    avg = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for it, (feats, coords, centers, mask) in enumerate(pbar):
        feats = feats.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # LR warmup
        lr = linear_warmup(global_step, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        ctx = torch.cuda.amp.autocast(enabled=args.amp) if args.amp else nullcontext()
        with ctx:
            rec, curv_pred, logits, _post = model(feats, centers, mask, tau=args.tau, hard=True)

            B, T, K, _ = rec.shape
            m = mask.reshape(B * T)
            if m.any():
                rec_m = rec.reshape(B * T, K, 3)[m]
                tgt_m = coords.reshape(B * T, K, 3)[m]
                cd = chamfer_distance(rec_m, tgt_m)

                curv_t = curvature_proxy(tgt_m).detach()
                curv_p = curv_pred.reshape(B * T)[m]
                l_curv = F.mse_loss(curv_p, curv_t)

                C = logits.shape[-1]
                q = torch.softmax(logits / max(1e-4, args.tau), dim=-1)
                q_m = q.reshape(B * T, C)[m]
                logC = math.log(C)
                kl = (q_m.clamp_min(1e-9).log().mul(q_m).sum(dim=-1) + logC).mean()
            else:
                cd = rec.mean() * 0.0
                l_curv = rec.mean() * 0.0
                kl = rec.mean() * 0.0

            loss = cd + args.curv_weight * l_curv + args.kl_weight * kl

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

        avg = 0.9 * avg + 0.1 * float(loss.item()) if it > 0 else float(loss.item())
        pbar.set_postfix({
            "lr": f"{lr:.2e}",
            "loss": f"{avg:.4f}",
            "cd": f"{float(cd.item()):.4f}",
            "kl": f"{float(kl.item()):.4f}",
        })
        global_step += 1

    if scheduler is not None:
        scheduler.step(avg)
    return avg, global_step


def save_ckpt(state: dict, save_dir: str, tag: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ckpt_{tag}.pt")
    torch.save(state, path)
    return path


def main():
    ap = argparse.ArgumentParser("Stage-2 Surface-VQMAE pretraining (PDBBind)")

    ap.add_argument("--data_root", type=str, required=True, help="Directory containing *.npz (one kind)")
    ap.add_argument("--save_dir", type=str, required=True, help="Directory to save checkpoints")

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
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default="")

    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--random_rotate", action="store_true", help="SO(3) augmentation for rel_xyz+normals")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    assert os.path.isdir(args.data_root), f"data_root not found: {args.data_root}"

    ds = SurfacePatchDataset(
        args.data_root,
        seq_len=args.seq_len,
        K=args.K,
        mask_ratio=args.mask_ratio,
        random_rotate=args.random_rotate,
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
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    start_epoch = 0
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt.get("optim", optimizer.state_dict()))
        if args.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[Resume] Loaded from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        # linear anneal tau -> tau_min across epochs
        p = epoch / max(1, args.epochs - 1)
        tau_now = args.tau * (1 - p) + args.tau_min * p
        args.tau = float(tau_now)

        loss, global_step = train_one_epoch(
            model,
            dl,
            optimizer,
            scaler,
            device,
            epoch,
            args,
            scheduler=scheduler,
            global_step=global_step,
        )

        if (epoch + 1) % args.save_every == 0:
            path = save_ckpt(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "global_step": global_step,
                    "args": vars(args),
                },
                args.save_dir,
                f"e{epoch:03d}",
            )
            print(f"[Save] {path}")

    path = save_ckpt(
        {
            "epoch": args.epochs - 1,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
            "global_step": global_step,
            "args": vars(args),
        },
        args.save_dir,
        "final",
    )
    print(f"[Done] saved to {path}")


if __name__ == "__main__":
    main()
