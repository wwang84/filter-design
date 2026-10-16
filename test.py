#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py - Evaluation script: pure S-parameter input, full-band single model
Auto-detects model layer counts from checkpoint. Includes tapered_line-aware statistics.
"""

import argparse, json, random, re
from pathlib import Path
import numpy as np, torch

from dataset import (
    load_all_subdevices, compute_cascade, flatten_device_s_to_tensor,
    _compute_adjacent_width_diffs
)
from model import ResidualNet
from utils import get_device


def build_prediction_input(S_subs, stats, freqs, port_widths=None):
    """
    Build model input, supports 8 or 9 channels (with port width difference).
    Consistent with predict.py build_prediction_input.
    """
    n_freq = len(freqs)
    num_dev = len(S_subs)
    max_len = stats.get('max_len', 6)
    freq_min = stats.get('freq_min', 0.5)
    freq_max = stats.get('freq_max', 15.0)
    freq_norm = ((freqs - freq_min) / (freq_max - freq_min + 1e-8)).astype(np.float32)

    dm = np.array(stats.get('device_s_mean', [0]*8))
    ds = np.array(stats.get('device_s_std', [1]*8))
    n_ch = len(dm)

    devices_s = np.zeros((max_len, n_ch, n_freq), dtype=np.float32)
    for i, S_sub in enumerate(S_subs):
        if i >= max_len:
            break
        s_tensor_8ch = flatten_device_s_to_tensor(S_sub, n_freq)
        for ch in range(8):
            s_tensor_8ch[ch] = (s_tensor_8ch[ch] - dm[ch]) / ds[ch]
        devices_s[i, :8] = s_tensor_8ch
        if n_ch >= 9:
            pw = float(port_widths[i]) if port_widths and i < len(port_widths) else 0.0
            devices_s[i, 8] = (pw - dm[8]) / ds[8]

    mask = np.zeros(max_len, dtype=bool)
    mask[:num_dev] = True

    return {
        'devices_s': torch.from_numpy(devices_s).unsqueeze(0),
        'mask': torch.from_numpy(mask).unsqueeze(0),
        'freq_norm': torch.from_numpy(freq_norm).unsqueeze(0)
    }


def denormalize_prediction(pred_norm, stats, n_freq):
    if 'target_mean_per_channel' in stats and 'target_std_per_channel' in stats:
        mean_per_ch = np.array(stats['target_mean_per_channel'])
        std_per_ch = np.array(stats['target_std_per_channel'])
        n_ch = len(pred_norm) // n_freq
        pred_reshaped = pred_norm.reshape(n_ch, n_freq)
        for ch in range(n_ch):
            pred_reshaped[ch] = pred_reshaped[ch] * std_per_ch[ch] + mean_per_ch[ch]
        return pred_reshaped.reshape(-1)
    return pred_norm


def compute_detailed_errors(S_pred, S_true, freqs):
    ports = [(0,0),(1,0),(1,1),(0,1)]
    port_names = ['S11','S21','S22','S12']
    bands = [
        freqs <= 4.5,
        (freqs > 4.0) & (freqs <= 8.5),
        (freqs > 8.0) & (freqs <= 12.5),
        freqs > 12.0
    ]
    band_labels = ['0.5-4.5', '4.0-8.5', '8.0-12.5', '12.0-15.0']

    results = {}
    all_errors = []
    all_band_errors = [[], [], [], []]

    for p_idx, (p, q) in enumerate(ports):
        mag_pred = np.abs(S_pred[:, p, q]) + 1e-12
        mag_true = np.abs(S_true[:, p, q]) + 1e-12
        db_pred = 20 * np.log10(mag_pred)
        db_true = 20 * np.log10(mag_true)
        err = np.abs(db_pred - db_true)
        results[port_names[p_idx]] = {
            'mae': np.mean(err),
            'mae_by_band': [np.mean(err[b]) if np.any(b) else 0 for b in bands],
        }
        all_errors.extend(err)
        for bi, b in enumerate(bands):
            if np.any(b):
                all_band_errors[bi].extend(err[b])

    all_errors = np.array(all_errors)
    results['total'] = {
        'mae': np.mean(all_errors),
        'mae_by_band': [np.mean(np.array(e)) if e else 0 for e in all_band_errors],
    }
    return results


def get_max_layers_from_keys(keys, prefix):
    """Extract max layer index + 1 from state dict keys for a given prefix"""
    max_idx = -1
    pattern = re.compile(rf'{prefix}\.layers\.(\d+)')
    for k in keys:
        m = pattern.search(k)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else None


def load_model_auto(model_dir, device):
    """Auto-detect checkpoint and create corresponding model instance"""
    model_dir = Path(model_dir)
    model_path = model_dir / "models" / "best_model.pth"
    stats_path = model_dir / "models" / "best_model_stats.json"
    if not model_path.exists():
        model_path = model_dir / "best_model.pth"
        stats_path = model_dir / "best_model_stats.json"
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}")

    with open(stats_path) as f:
        stats = json.load(f)
    n_freq = stats['n_freq']
    max_len = stats.get('max_len', 6)

    ckpt = torch.load(model_path, map_location='cpu')
    raw_keys = [k[7:] if k.startswith('module.') else k for k in ckpt.keys()]

    freq_layers = get_max_layers_from_keys(raw_keys, 'freq_self_attn')
    device_layers = get_max_layers_from_keys(raw_keys, 'device_cross_attn')
    if freq_layers is None or device_layers is None:
        freq_layers = 4
        device_layers = 4

    print(f"  Auto-detected: freq_layers={freq_layers}, device_layers={device_layers}")

    model = ResidualNet(
        n_freq=n_freq,
        max_len=max_len,
        d_model=320,
        nhead=8,
        num_device_layers=device_layers,
        num_freq_layers=freq_layers,
        dim_ff=768,
        dropout=0.15,
        attn_dropout=0.05
    )

    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"  Model loaded: {n_freq} freq points, {sum(p.numel() for p in model.parameters()):,} params")
    return model, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_mode", type=str, default="db_phase", choices=["ri", "db_phase"])
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, stats = load_model_auto(args.model_dir, device)
    n_freq = stats['n_freq']

    data_dir = Path(args.data_dir)
    sub_map = load_all_subdevices(data_dir)
    print(f"Loaded {len(sub_map)} subdevice entries")
    with open(data_dir / "cascade" / "train.json") as f:
        cascade_data = json.load(f)
    total_samples = len(cascade_data)
    print(f"Total cascade samples: {total_samples}")

    random.seed(args.seed)
    num_to_sample = min(args.num_samples, total_samples)
    sample_indices = random.sample(range(total_samples), num_to_sample)
    print(f"Randomly sampled {num_to_sample} samples")

    cas_records = []
    cor_records = []
    sample_ids = []
    seq_lengths = []
    contains_tapered = []

    for idx in sample_indices:
        item = cascade_data[idx]
        seq_info = item['sequence']
        sample_n = item['n']

        S_subs = []
        geoms = []
        freqs = None
        for fam, n in seq_info:
            key = (fam, n)
            if key not in sub_map:
                continue
            sub = sub_map[key]
            S_subs.append(sub['S'])
            geoms.append(sub['geom'])
            if freqs is None:
                freqs = sub['freqs']
        if freqs is None:
            continue

        port_widths = _compute_adjacent_width_diffs(seq_info, geoms)
        S_cascade = compute_cascade(S_subs)

        y_true = np.array(item['s'])
        S_true = np.zeros((len(y_true), 2, 2), dtype=complex)
        for i, row in enumerate(y_true):
            S_true[i, 0, 0] = complex(row[1], row[2])
            S_true[i, 1, 0] = complex(row[3], row[4])
            S_true[i, 1, 1] = complex(row[5], row[6])
            S_true[i, 0, 1] = complex(row[7], row[8])

        err_cas = compute_detailed_errors(S_cascade, S_true, freqs)

        inputs = build_prediction_input(S_subs, stats, freqs, port_widths=port_widths)
        with torch.no_grad():
            devices_s = inputs['devices_s'].to(device)
            mask = inputs['mask'].to(device)
            freq_norm = inputs['freq_norm'].to(device)
            pred_norm = model(devices_s, mask, freq_norm).squeeze(0).cpu().numpy()

        pred_db = denormalize_prediction(pred_norm, stats, n_freq).reshape(4, n_freq)

        S_corrected = np.zeros((len(freqs), 2, 2), dtype=complex)
        ports = [(0,0),(1,0),(1,1),(0,1)]
        for p_idx, (p, q) in enumerate(ports):
            mag = 10**(pred_db[p_idx, :] / 20.0)
            S_corrected[:, p, q] = mag * np.exp(1j * 0)

        err_cor = compute_detailed_errors(S_corrected, S_true, freqs)

        cas_records.append(err_cas)
        cor_records.append(err_cor)
        sample_ids.append(sample_n)
        seq_lengths.append(len(seq_info))
        has_tapered = any(fam == 'tapered_line' for fam, _ in seq_info)
        contains_tapered.append(has_tapered)

    total = len(cor_records)

    band_labels = ['0.5-4.5', '4.0-8.5', '8.0-12.5', '12.0-15.0']

    print("\n" + "=" * 80)
    print("A. Grouped by number of devices")
    print("=" * 80)
    print(f"{'Devs':>5s} {'N':>5s} {'Cascade':>8s} {'Model':>8s} {'Improve':>8s}  {'S11_cas':>8s} {'S11_mod':>8s}  {'S21_cas':>8s} {'S21_mod':>8s}")
    for n_dev in sorted(set(seq_lengths)):
        idxs = [i for i, l in enumerate(seq_lengths) if l == n_dev]
        if not idxs:
            continue
        cm = np.mean([cas_records[i]['total']['mae'] for i in idxs])
        mm = np.mean([cor_records[i]['total']['mae'] for i in idxs])
        s11c = np.mean([cas_records[i]['S11']['mae'] for i in idxs])
        s11m = np.mean([cor_records[i]['S11']['mae'] for i in idxs])
        s21c = np.mean([cas_records[i]['S21']['mae'] for i in idxs])
        s21m = np.mean([cor_records[i]['S21']['mae'] for i in idxs])
        print(f"{n_dev:>5d} {len(idxs):>5d} {cm:>8.4f} {mm:>8.4f} {cm-mm:>8.4f}  {s11c:>8.4f} {s11m:>8.4f}  {s21c:>8.4f} {s21m:>8.4f}")

    print("\n" + "=" * 80)
    print("B. Grouped by frequency band")
    print("=" * 80)
    print(f"{'Port':>6s}  {'Cascade':>8s} {'Model':>8s} {'Improve':>8s}  |{'0.5-4.5':>10s} {'4.0-8.5':>10s} {'8.0-12.5':>10s} {'12.0-15.0':>10s}")
    for port in ['total', 'S11', 'S21', 'S22', 'S12']:
        cm_all = np.mean([r[port]['mae'] for r in cas_records])
        mm_all = np.mean([r[port]['mae'] for r in cor_records])
        mm_bands = np.mean([r[port]['mae_by_band'] for r in cor_records], axis=0)
        print(f"{port:>6s}  {cm_all:>8.4f} {mm_all:>8.4f} {cm_all-mm_all:>8.4f}  |{mm_bands[0]:>10.4f} {mm_bands[1]:>10.4f} {mm_bands[2]:>10.4f} {mm_bands[3]:>10.4f}")

    print("\n" + "=" * 80)
    print("C. Cross-analysis: devices x band (Model MAE)")
    print("=" * 80)
    header = f"{'Devs':>5s}  " + "  ".join([f"{l:>10s}" for l in band_labels])
    print(header)
    for n_dev in sorted(set(seq_lengths)):
        idxs = [i for i, l in enumerate(seq_lengths) if l == n_dev]
        if len(idxs) < 5:
            continue
        band_mae = np.mean([cor_records[i]['total']['mae_by_band'] for i in idxs], axis=0)
        row = f"{n_dev:>5d}  " + "  ".join([f"{v:>10.4f}" for v in band_mae])
        print(row)

    print("\n" + "=" * 80)
    print("D. Grouped by tapered_line presence")
    print("=" * 80)
    idx_tapered = [i for i, flag in enumerate(contains_tapered) if flag]
    idx_non = [i for i, flag in enumerate(contains_tapered) if not flag]
    print(f"Samples with tapered_line: {len(idx_tapered)}")
    print(f"Samples without tapered_line: {len(idx_non)}")

    if len(idx_tapered) == 0:
        print("Warning: No samples with tapered_line found.")
    else:
        print("\n[Overall MAE]")
        print(f"{'Group':>15s} {'Cascade':>10s} {'Model':>10s} {'Improve':>10s}  {'S11_imp':>10s} {'S21_imp':>10s}")
        for group, idxs, label in [
            ('with tapered_line', idx_tapered, 'Y'),
            ('w/o tapered_line', idx_non, 'N')
        ]:
            if not idxs:
                continue
            cm = np.mean([cas_records[i]['total']['mae'] for i in idxs])
            mm = np.mean([cor_records[i]['total']['mae'] for i in idxs])
            s11c = np.mean([cas_records[i]['S11']['mae'] for i in idxs])
            s11m = np.mean([cor_records[i]['S11']['mae'] for i in idxs])
            s21c = np.mean([cas_records[i]['S21']['mae'] for i in idxs])
            s21m = np.mean([cor_records[i]['S21']['mae'] for i in idxs])
            imp_total = cm - mm
            imp_s11 = s11c - s11m
            imp_s21 = s21c - s21m
            print(f"{label + ' (' + group + ')':>15s} {cm:>10.4f} {mm:>10.4f} {imp_total:>10.4f}  {imp_s11:>10.4f} {imp_s21:>10.4f}")

        print("\n[Model MAE by band]")
        print(f"{'Group':>15s}  {'0.5-4.5':>10s} {'4.0-8.5':>10s} {'8.0-12.5':>10s} {'12.0-15.0':>10s}")
        for group, idxs in [('with tapered_line', idx_tapered), ('w/o tapered_line', idx_non)]:
            if not idxs:
                continue
            band_mae = np.mean([cor_records[i]['total']['mae_by_band'] for i in idxs], axis=0)
            row = f"{group:>15s}  " + "  ".join([f"{v:>10.4f}" for v in band_mae])
            print(row)

        print("\n[Improvement rate]")
        for group, idxs in [('with tapered_line', idx_tapered), ('w/o tapered_line', idx_non)]:
            if not idxs:
                continue
            improved = sum(cas_records[i]['total']['mae'] > cor_records[i]['total']['mae'] for i in idxs)
            print(f"{group:>15s}: {improved}/{len(idxs)} ({100*improved/len(idxs):.1f}%)")

    improved_all = sum(cas_records[i]['total']['mae'] > cor_records[i]['total']['mae'] for i in range(total))
    print(f"\nOverall improvement rate: {improved_all}/{total} ({100 * improved_all / total:.1f}%)")
    print(f"Overall cascade MAE: {np.mean([r['total']['mae'] for r in cas_records]):.4f} -> Overall model MAE: {np.mean([r['total']['mae'] for r in cor_records]):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
