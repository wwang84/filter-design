#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_block.py

HFSS segmented filter subdevice forward simulation surrogate model training script.

Features:
- One script for all subdevice types
- Reads dataset/<family>/train.json
- Trains x -> y forward model
- Each subdevice model saved to checkpoints/<family>/
- Saves: model.pt, config.json, loss.csv, norm_stats.pt
1
Input:
    x = [f_ghz, geometry...]

Output:
    y = [s11_re, s11_im, s21_re, s21_im, s22_re, s22_im, s12_re, s12_im]

Includes:
- Fourier feature encoding
- Residual MLP
- AdamW
- CosineAnnealingLR
- Validation split
- Early stopping
- Passivity penalty
- Reciprocity penalty
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Basic configuration
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
CKPT_DIR = ROOT_DIR / "checkpoints"

FAMILIES = ["single_step", "open_stub", "t_stub", "double_step", "tapered_line"]

DEFAULT_EPOCHS = 15000
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIDDEN = 256
DEFAULT_DEPTH = 6
DEFAULT_FOURIER_BANDS = 16
DEFAULT_VAL_RATIO = 0.15
DEFAULT_PATIENCE = 1500

SEED = 2026


# ============================================================
# Utility functions
# ============================================================
def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise RuntimeError(f"{path} is empty")

    data = json.loads(text)

    if not isinstance(data, list):
        raise RuntimeError(f"{path} top-level is not a JSON array")

    return data


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_config(path: Path, cfg: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def append_loss_csv(path: Path, row: dict, write_header: bool = False):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_mse",
        "val_mse",
        "train_passivity",
        "val_passivity",
        "train_reciprocity",
        "val_reciprocity",
        "lr",
    ]

    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header or not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Dataset
# ============================================================
class SParamDataset(Dataset):
    def __init__(self, train_json: Path):
        raw = load_json_array(train_json)

        x_all = []
        y_all = []

        x_order = None
        y_order = None

        for obj in raw:
            if "x" not in obj or "y" not in obj:
                raise RuntimeError(f"{train_json} contains sample missing x/y: n={obj.get('n')}")

            if x_order is None:
                x_order = obj.get("x_order")

            if y_order is None:
                y_order = obj.get("y_order")

            x = obj["x"]
            y = obj["y"]

            if len(x) != len(y):
                raise RuntimeError(f"Sample n={obj.get('n')} has mismatched x/y lengths")

            x_all.extend(x)
            y_all.extend(y)

        self.x_order = x_order
        self.y_order = y_order

        self.x = torch.tensor(x_all, dtype=torch.float32)
        self.y = torch.tensor(y_all, dtype=torch.float32)

        if self.y.shape[1] != 8:
            raise RuntimeError(f"y dimension should be 8, got {self.y.shape[1]}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def compute_norm_stats(x: torch.Tensor, y: torch.Tensor):
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-8)

    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0).clamp_min(1e-8)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


# ============================================================
# Fourier Feature Encoding
# ============================================================
class FourierFeatureEncoder(nn.Module):
    """
    Fourier feature encoding for normalized inputs.

    Input dimension:
        input_dim

    Output dimension:
        input_dim + input_dim * bands * 2

    All input dimensions are encoded, including f_ghz and geometry parameters.
    Fourier encoding of f_ghz is especially useful for S-parameter data
    which varies significantly with frequency.
    """
    def __init__(self, input_dim: int, bands: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.bands = bands

        freq_bands = 2.0 ** torch.arange(bands, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

        self.out_dim = input_dim + input_dim * bands * 2

    def forward(self, x):
        # x: [B, input_dim]
        xb = x.unsqueeze(-1) * self.freq_bands  # [B, input_dim, bands]
        xb = math.pi * xb

        sin_feat = torch.sin(xb).flatten(start_dim=1)
        cos_feat = torch.cos(xb).flatten(start_dim=1)

        return torch.cat([x, sin_feat, cos_feat], dim=1)


# ============================================================
# Residual MLP
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SParamForwardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 8,
        hidden_dim: int = 256,
        depth: int = 6,
        fourier_bands: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.encoder = FourierFeatureEncoder(
            input_dim=input_dim,
            bands=fourier_bands,
        )

        self.in_proj = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_norm):
        z = self.encoder(x_norm)
        z = self.in_proj(z)
        z = self.blocks(z)
        y_norm = self.out(z)
        return y_norm


# ============================================================
# Physics soft constraints
# ============================================================
def unnormalize_y(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def passivity_penalty(y_real):
    """
    y_real:
    [s11_re, s11_im, s21_re, s21_im, s22_re, s22_im, s12_re, s12_im]

    For a passive 2-port network, approximate constraint:
        |S11|^2 + |S21|^2 <= 1
        |S22|^2 + |S12|^2 <= 1

    Applied as a soft penalty, not hard clipping.
    """
    s11_re = y_real[:, 0]
    s11_im = y_real[:, 1]
    s21_re = y_real[:, 2]
    s21_im = y_real[:, 3]
    s22_re = y_real[:, 4]
    s22_im = y_real[:, 5]
    s12_re = y_real[:, 6]
    s12_im = y_real[:, 7]

    s11_abs2 = s11_re ** 2 + s11_im ** 2
    s21_abs2 = s21_re ** 2 + s21_im ** 2
    s22_abs2 = s22_re ** 2 + s22_im ** 2
    s12_abs2 = s12_re ** 2 + s12_im ** 2

    p1 = F.relu(s11_abs2 + s21_abs2 - 1.0)
    p2 = F.relu(s22_abs2 + s12_abs2 - 1.0)

    return (p1.mean() + p2.mean()) * 0.5


def reciprocity_penalty(y_real):
    """
    For a passive reciprocal 2-port, S21 approx S12.
    Applied as a soft penalty.
    """
    s21_re = y_real[:, 2]
    s21_im = y_real[:, 3]
    s12_re = y_real[:, 6]
    s12_im = y_real[:, 7]

    return ((s21_re - s12_re) ** 2 + (s21_im - s12_im) ** 2).mean()


# ============================================================
# Training and validation
# ============================================================
def run_one_epoch(
    model,
    loader,
    optimizer,
    device,
    norm_stats,
    train: bool,
    passivity_weight: float,
    reciprocity_weight: float,
):
    if train:
        model.train()
    else:
        model.eval()

    x_mean = norm_stats["x_mean"].to(device)
    x_std = norm_stats["x_std"].to(device)
    y_mean = norm_stats["y_mean"].to(device)
    y_std = norm_stats["y_std"].to(device)

    total_loss = 0.0
    total_mse = 0.0
    total_passivity = 0.0
    total_reciprocity = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        with torch.set_grad_enabled(train):
            pred_norm = model(x_norm)

            mse = F.mse_loss(pred_norm, y_norm)

            pred_real = unnormalize_y(pred_norm, y_mean, y_std)

            p_loss = passivity_penalty(pred_real)
            r_loss = reciprocity_penalty(pred_real)

            loss = mse + passivity_weight * p_loss + reciprocity_weight * r_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        bs = x.shape[0]

        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_passivity += p_loss.item() * bs
        total_reciprocity += r_loss.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "mse": total_mse / total_n,
        "passivity": total_passivity / total_n,
        "reciprocity": total_reciprocity / total_n,
    }


def train_family(args, family: str):
    train_json = DATASET_DIR / family / "train.json"
    save_dir = CKPT_DIR / family

    ensure_dir(save_dir)

    loss_csv = save_dir / "loss.csv"
    if loss_csv.exists():
        loss_csv.unlink()

    print("=" * 80)
    print(f"Training family: {family}")
    print(f"Data: {train_json}")
    print(f"Save: {save_dir}")

    dataset = SParamDataset(train_json)

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    norm_stats = compute_norm_stats(dataset.x, dataset.y)

    input_dim = dataset.x.shape[1]
    output_dim = dataset.y.shape[1]

    device = get_device()

    model = SParamForwardModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden,
        depth=args.depth,
        fourier_bands=args.fourier_bands,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.02,
    )

    cfg = {
        "family": family,
        "train_json": str(train_json),
        "x_order": dataset.x_order,
        "y_order": dataset.y_order,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden": args.hidden,
        "depth": args.depth,
        "fourier_bands": args.fourier_bands,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "passivity_weight": args.passivity_weight,
        "reciprocity_weight": args.reciprocity_weight,
        "n_total_points": n_total,
        "n_train_points": n_train,
        "n_val_points": n_val,
        "device": str(device),
    }

    write_config(save_dir / "config.json", cfg)
    torch.save(norm_stats, save_dir / "norm_stats.pt")

    best_val = float("inf")
    best_epoch = 0
    bad_count = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            norm_stats=norm_stats,
            train=True,
            passivity_weight=args.passivity_weight,
            reciprocity_weight=args.reciprocity_weight,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            norm_stats=norm_stats,
            train=False,
            passivity_weight=args.passivity_weight,
            reciprocity_weight=args.reciprocity_weight,
        )

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        append_loss_csv(
            loss_csv,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "train_passivity": train_metrics["passivity"],
                "val_passivity": val_metrics["passivity"],
                "train_reciprocity": train_metrics["reciprocity"],
                "val_reciprocity": val_metrics["reciprocity"],
                "lr": lr_now,
            },
            write_header=(epoch == 1),
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            bad_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "norm_stats": norm_stats,
                    "config": cfg,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                save_dir / "model.pt",
            )

        else:
            bad_count += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[{family}] "
                f"epoch={epoch:04d} "
                f"train={train_metrics['loss']:.6e} "
                f"val={val_metrics['loss']:.6e} "
                f"mse={val_metrics['mse']:.6e} "
                f"pass={val_metrics['passivity']:.3e} "
                f"recip={val_metrics['reciprocity']:.3e} "
                f"lr={lr_now:.3e} "
                f"best={best_val:.6e}@{best_epoch}"
            )

        if bad_count >= args.patience:
            print(
                f"[{family}] early stop at epoch {epoch}, "
                f"best val={best_val:.6e} at epoch {best_epoch}"
            )
            break

    print(f"[{family}] finished. best val={best_val:.6e} at epoch {best_epoch}")
    print(f"[{family}] saved to {save_dir / 'model.pt'}")


# ============================================================
# Inference functions - callable from optimization scripts
# ============================================================
def load_trained_model(model_path: Path, device=None):
    if device is None:
        device = get_device()

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]

    model = SParamForwardModel(
        input_dim=cfg["input_dim"],
        output_dim=cfg["output_dim"],
        hidden_dim=cfg["hidden"],
        depth=cfg["depth"],
        fourier_bands=cfg["fourier_bands"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_stats = ckpt["norm_stats"]

    return model, norm_stats, cfg


@torch.no_grad()
def predict_sparams(model, norm_stats, x_numpy_or_tensor, device=None):
    """
    x_numpy_or_tensor:
        shape = [N, input_dim]

    Returns:
        y_pred, shape = [N, 8]
    """
    if device is None:
        device = get_device()

    if not torch.is_tensor(x_numpy_or_tensor):
        x = torch.tensor(x_numpy_or_tensor, dtype=torch.float32)
    else:
        x = x_numpy_or_tensor.float()

    x = x.to(device)

    x_mean = norm_stats["x_mean"].to(device)
    x_std = norm_stats["x_std"].to(device)
    y_mean = norm_stats["y_mean"].to(device)
    y_std = norm_stats["y_std"].to(device)

    x_norm = (x - x_mean) / x_std
    y_norm = model(x_norm)
    y = unnormalize_y(y_norm, y_mean, y_std)

    return y.cpu()


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        nargs='*',
        default=["all"],
        help="Subdevice types to train. Can specify multiple, e.g., single_step open_stub; or 'all' to train all",
    )

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--fourier_bands", type=int, default=DEFAULT_FOURIER_BANDS)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--print_every", type=int, default=10)

    parser.add_argument("--passivity_weight", type=float, default=0.02)
    parser.add_argument("--reciprocity_weight", type=float, default=0.02)

    return parser.parse_args()


def main():
    set_seed(SEED)
    args = parse_args()

    families = args.family
    if "all" in families:
        if len(families) > 1:
            print("Warning: 'all' specified with other devices, training all.")
        families_to_train = FAMILIES
    else:
        invalid = [f for f in families if f not in FAMILIES]
        if invalid:
            raise ValueError(f"Unknown device name(s): {', '.join(invalid)}. Available: {FAMILIES}")
        families_to_train = families

    print(f"Subdevices to train: {families_to_train}")
    for family in families_to_train:
        train_family(args, family)


if __name__ == "__main__":
    main()
