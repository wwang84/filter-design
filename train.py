#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - Training script (9ch RI + port_width_diff)

Requires precomputed S-parameter cache (run precompute_predicted_sparams.py first)
to use --use_predicted_sparams for distribution-shift-aware training.
"""

import argparse
import time
import warnings
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim

from dataset import (
    load_all_subdevices,
    load_cascade_data,
    load_predicted_sparams_cache,
    build_dataloaders_by_n_split,
)
from model import ResidualNet
from loss import TotalLoss
from utils import (
    TrainLogger, set_seed, count_parameters, ensure_dir,
    format_time, save_model_with_stats,
)

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cascade Correction Model (9ch)')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--log_dir', type=str, default='./logs/residual',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_mode', type=str, default='db_phase',
                        choices=['ri', 'db_phase'])
    parser.add_argument('--device_weight_power', type=float, default=1.0)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--inv_var_power', type=float, default=0.5)
    parser.add_argument('--huber_beta', type=float, default=1.0)
    parser.add_argument('--use_predicted_sparams', action='store_true',
                        help='Train with predicted S-parameter cache (run precompute_predicted_sparams.py first)')
    parser.add_argument('--predicted_cache', type=str, default='./predicted_sparams_cache.npz',
                        help='Path to predicted S-parameter cache')
    parser.add_argument('--noise_std_pred', type=float, default=0.03)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--d_model', type=int, default=320,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_device_layers', type=int, default=4,
                        help='Number of device cross-attention layers')
    parser.add_argument('--num_freq_layers', type=int, default=4,
                        help='Number of frequency self-attention layers')
    parser.add_argument('--dim_ff', type=int, default=768,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--attn_dropout', type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Full-band training (0.5-15 GHz, 9ch: 8ch RI + port_width_diff)")

    data_dir = Path(args.data_dir)
    print("Loading subdevice data...")
    sub_map = load_all_subdevices(data_dir)
    print(f"Loaded {len(sub_map)} subdevice entries")

    predicted_cache = None
    noise_std_augment = 0.02
    if args.use_predicted_sparams:
        cache_path = Path(args.predicted_cache)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"predicted_sparams_cache.npz not found at {cache_path}. "
                f"Run precompute_predicted_sparams.py first to generate it."
            )
        predicted_cache = load_predicted_sparams_cache(cache_path)
        print(f"  loaded {len(predicted_cache)} predicted sparam entries")
        noise_std_augment = args.noise_std_pred

    print("Loading cascade data...")
    cascade_data = load_cascade_data(
        data_dir / "cascade", sub_map,
        predicted_sparams_cache=predicted_cache,
    )
    print(f"Loaded {len(cascade_data)} cascade samples")

    train_loader, val_loader, train_ds, val_ds, stats = build_dataloaders_by_n_split(
        cascade_data, output_mode=args.output_mode, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=True, noise_std=noise_std_augment,
        freq_min=None, freq_max=None, max_len=6,
        train_ratio=args.train_ratio, random_state=args.seed,
        n_selected=0, freq_range=None,
    )

    n_freq = train_ds.n_freq
    print(f"n_freq: {n_freq}")

    model = ResidualNet(
        n_freq=n_freq, max_len=6,
        d_model=args.d_model, nhead=args.nhead,
        num_device_layers=args.num_device_layers,
        num_freq_layers=args.num_freq_layers,
        dim_ff=args.dim_ff, dropout=args.dropout,
        attn_dropout=args.attn_dropout,
    ).to(device)

    if torch.cuda.device_count() > 1 and args.gpu == 0:
        model = torch.nn.DataParallel(model)
    print(f"Model params: {count_parameters(model):,}")

    channel_std = torch.tensor(train_ds.target_std_per_channel, dtype=torch.float32).to(device)
    channel_mean = torch.tensor(train_ds.target_mean_per_channel, dtype=torch.float32).to(device)
    freq_std_target = stats.get('freq_std_target')
    freq_std = torch.tensor(freq_std_target, dtype=torch.float32).to(device) if freq_std_target else None

    loss_fn = TotalLoss(huber_beta=args.huber_beta, amplitude_weight_strength=2.0,
                          reciprocal_weight=0.2, smooth_weight=0.05,
                          channel_std=channel_std, channel_mean=channel_mean,
                          freq_std=freq_std, inv_var_power=args.inv_var_power,
                          device_weight_power=args.device_weight_power)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, cooldown=5, min_lr=1e-6)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_loss', float('inf'))

    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    model_dir = log_dir / 'models'; ensure_dir(model_dir)
    checkpoint_dir = log_dir / 'checkpoints'; ensure_dir(checkpoint_dir)
    logger = TrainLogger(log_dir)
    scaler = torch.amp.GradScaler(device.type, enabled=args.use_amp) if device.type == 'cuda' else None
    patience = 80; counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = train_db_mae = train_loss_rec = train_loss_smooth = 0.0
        t0 = time.time()
        for batch in train_loader:
            devices_s, mask, freq_norm, target = [x.to(device) for x in batch]
            optimizer.zero_grad()
            if args.use_amp and device.type == 'cuda':
                with torch.amp.autocast(device.type, enabled=True):
                    pred = model(devices_s, mask, freq_norm)
                    total_loss, db_mae, loss_rec, loss_smooth, _ = loss_fn(pred, target, n_freq, None, None)
                scaler.scale(total_loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(devices_s, mask, freq_norm)
                total_loss, db_mae, loss_rec, loss_smooth, _ = loss_fn(pred, target, n_freq, None, None)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = devices_s.size(0)
            train_loss += total_loss.item() * bs
            train_db_mae += db_mae.item() * bs
            train_loss_rec += loss_rec.item() * bs
            train_loss_smooth += loss_smooth.item() * bs
        n = len(train_loader.dataset)
        train_loss /= n; train_db_mae /= n; train_loss_rec /= n; train_loss_smooth /= n

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                devices_s, mask, freq_norm, target = [x.to(device) for x in batch]
                if args.use_amp and device.type == 'cuda':
                    with torch.amp.autocast(device.type, enabled=True):
                        pred = model(devices_s, mask, freq_norm)
                        tl, _, _, _, _ = loss_fn(pred, target, n_freq, None, None)
                else:
                    pred = model(devices_s, mask, freq_norm)
                    tl, _, _, _, _ = loss_fn(pred, target, n_freq, None, None)
                val_loss += tl.item() * devices_s.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        if val_loss < best_val_loss:
            best_val_loss = val_loss; counter = 0
            save_stats = {
                'epoch': epoch, 'best_val_loss': best_val_loss, 'n_freq': n_freq, 'max_len': 6,
                'freq_min': float(train_ds.freq_min), 'freq_max': float(train_ds.freq_max),
                'target_mean_per_channel': train_ds.target_mean_per_channel.tolist(),
                'target_std_per_channel': train_ds.target_std_per_channel.tolist(),
                'device_s_mean': train_ds.device_s_mean.tolist(),
                'device_s_std': train_ds.device_s_std.tolist(),
                'freq_std_target': freq_std.cpu().tolist() if freq_std is not None else None,
            }
            save_model_with_stats(model, save_stats, model_dir / 'best_model.pth', model_dir / 'best_model_stats.json')
            print(f"  Best model saved (val_loss={val_loss:.6f})")
        else:
            counter += 1
        if (epoch + 1) % 20 == 0 or epoch == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_loss': best_val_loss, 'train_loss': train_loss, 'val_loss': val_loss,
                        'config': vars(args)}, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth')
        logger.log(epoch, train_loss, val_loss, lr, best_val_loss)
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | LR: {lr:.2e} | "
              f"Train: {train_loss:.4f} (dB_MAE={train_db_mae:.4f}) | "
              f"Val: {val_loss:.4f} | Best: {best_val_loss:.4f} | {format_time(time.time() - t0)}")
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}"); break

    logger.plot_loss(save_path=log_dir / 'loss_curve.png')
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {model_dir}")


if __name__ == '__main__':
    main()
