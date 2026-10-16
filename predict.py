#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py - Prediction script: pure S-parameter input, full-band single model
Supports single sample or batch (range n-m) prediction with PDF generation.
"""

import argparse, json, warnings
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import (
    load_all_subdevices, flatten_device_s_to_tensor, load_predicted_sparams_cache,
    get_port_widths, _compute_adjacent_width_diffs,
)
from model import ResidualNet
from utils import get_device

warnings.filterwarnings('ignore')


def build_prediction_input(S_subs, stats, freqs, port_widths=None):
    n_freq = len(freqs); num_dev = len(S_subs); max_len = stats.get('max_len', 6)
    freq_min = stats.get('freq_min', 0.5); freq_max = stats.get('freq_max', 15.0)
    freq_norm = ((freqs - freq_min) / (freq_max - freq_min + 1e-8)).astype(np.float32)

    dm = np.array(stats.get('device_s_mean', [0]*9))
    ds = np.array(stats.get('device_s_std', [1]*9))
    n_ch = len(dm)
    devices_s = np.zeros((max_len, n_ch, n_freq), dtype=np.float32)
    for i, S_sub in enumerate(S_subs):
        if i >= max_len: break
        s_tensor_8ch = flatten_device_s_to_tensor(S_sub, n_freq)
        for ch in range(8):
            s_tensor_8ch[ch] = (s_tensor_8ch[ch] - dm[ch]) / ds[ch]
        devices_s[i, :8] = s_tensor_8ch
        if n_ch >= 9:
            pw = float(port_widths[i]) if port_widths and i < len(port_widths) else 0.0
            devices_s[i, 8] = (pw - dm[8]) / ds[8]

    mask = np.zeros(max_len, dtype=bool); mask[:num_dev] = True

    return {'devices_s': torch.from_numpy(devices_s).unsqueeze(0),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'freq_norm': torch.from_numpy(freq_norm).unsqueeze(0)}


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


def plot_comparison(freqs, db_true, db_corrected, seq_str, pdf_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    names = ["S11", "S21", "S22", "S12"]
    for ax, name, idx in zip(axes.flatten(), names, range(4)):
        ax.plot(freqs, db_corrected[:, idx], 'b-', label='Model', linewidth=1.5)
        if db_true is not None:
            ax.plot(freqs, db_true[:, idx], 'k-', label='True (HFSS)', linewidth=1.5, alpha=0.7)
            mae_cor = np.mean(np.abs(db_corrected[:, idx] - db_true[:, idx]))
            ax.set_title(f"{name}  MAE={mae_cor:.2f} dB")
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel(f"{name} (dB)")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(f"Prediction: {seq_str}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(pdf_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"PDF saved to {pdf_path}")


def parse_n_range(n_str):
    """Parse --n argument, returns (is_range, start, end) or (False, n, None)"""
    if '-' in n_str:
        parts = n_str.split('-')
        if len(parts) != 2:
            raise ValueError("Range format should be n-m, e.g., 10-20")
        try:
            start = int(parts[0]); end = int(parts[1])
        except ValueError:
            raise ValueError("Range values must be integers")
        if start > end:
            raise ValueError("Start value cannot be greater than end value")
        return True, start, end
    else:
        try:
            n = int(n_str)
        except ValueError:
            raise ValueError("--n must be a single integer or n-m range")
        return False, n, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=str, required=True,
                        help="Sample number (integer) or range (e.g., 10-20), generates PDF for all samples in range")
    parser.add_argument("--model_dir", type=str, default="./checkpoints/residual_v6_9ch",
                        help="Path to model directory")
    parser.add_argument("--output_pdf", type=str, default="./figures/predict.pdf",
                        help="Output PDF path for single sample")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for batch prediction")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Path to dataset directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--use_predicted_sparams", action="store_true",
                        help="Use predicted S-parameter cache instead of HFSS ground truth")
    parser.add_argument("--predicted_cache", type=str, default="./predicted_sparams_cache.npz",
                        help="Path to predicted S-parameter cache")
    args = parser.parse_args()

    is_range, n_start, n_end = parse_n_range(args.n)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_dir = Path(args.model_dir); data_dir = Path(args.data_dir)

    predicted_cache = None
    if args.use_predicted_sparams:
        cache_path = Path(args.predicted_cache)
        if not cache_path.exists():
            raise FileNotFoundError(f"Predicted cache not found: {cache_path}. Run precompute_predicted_sparams.py first.")
        print(f"Loading predicted S-param cache from {cache_path}...")
        predicted_cache = load_predicted_sparams_cache(cache_path)
        print(f"  loaded {len(predicted_cache)} predicted subdevice S-params")

    # Load model
    model_path = model_dir / "models" / "best_model.pth"
    stats_path = model_dir / "models" / "best_model_stats.json"
    if not model_path.exists():
        model_path = model_dir / "best_model.pth"
        stats_path = model_dir / "best_model_stats.json"

    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}")

    with open(stats_path) as f:
        stats = json.load(f)
    n_freq = stats['n_freq']; max_len = stats.get('max_len', 6)

    ckpt = torch.load(model_path, map_location='cpu')
    model = ResidualNet(n_freq=n_freq, max_len=max_len)
    # Strip 'module.' prefix (DataParallel)
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device); model.eval()
    print(f"Model loaded: {n_freq} freq points, {sum(p.numel() for p in model.parameters()):,} params")

    # Load data
    sub_map = load_all_subdevices(data_dir)
    with open(data_dir / "cascade" / "train.json") as f:
        cascade_json = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_range:
        n_list = list(range(n_start, n_end + 1))
        print(f"Batch prediction: samples {n_start}-{n_end}, total {len(n_list)}")
    else:
        n_list = [n_start]

    for n_target in n_list:
        target_item = None
        for item in cascade_json:
            if item['n'] == n_target:
                target_item = item
                break
        if target_item is None:
            print(f"Warning: sample {n_target} not found, skipping")
            continue

        seq_info = target_item['sequence']
        S_subs = []; freqs = None
        n_pred_used = 0
        for fam, n in seq_info:
            if predicted_cache is not None:
                key = f"{fam}_{n}"
                if key in predicted_cache:
                    y_pred = predicted_cache[key]
                    S = np.zeros((y_pred.shape[0], 2, 2), dtype=complex)
                    S[:, 0, 0] = y_pred[:, 0] + 1j * y_pred[:, 1]
                    S[:, 1, 0] = y_pred[:, 2] + 1j * y_pred[:, 3]
                    S[:, 1, 1] = y_pred[:, 4] + 1j * y_pred[:, 5]
                    S[:, 0, 1] = y_pred[:, 6] + 1j * y_pred[:, 7]
                    S_subs.append(S)
                    n_pred_used += 1
                    if freqs is None:
                        freqs = np.linspace(0.5, 15.0, y_pred.shape[0])
                    continue
            sub_data = sub_map[(fam, n)]
            S_subs.append(sub_data['S'])
            if freqs is None:
                freqs = sub_data['freqs']

        if predicted_cache is not None:
            print(f"  Sample {n_target}: using predicted S-params for {n_pred_used}/{len(seq_info)} subdevices")

        geoms = []
        for fam, n in seq_info:
            sd = sub_map[(fam, n)]
            geoms.append(sd['geom'])
        port_widths = _compute_adjacent_width_diffs(seq_info, geoms)

        y_true = np.array(target_item['s'])
        S_true = np.zeros((len(y_true), 2, 2), dtype=complex)
        for i, row in enumerate(y_true):
            S_true[i, 0, 0] = complex(row[1], row[2])
            S_true[i, 1, 0] = complex(row[3], row[4])
            S_true[i, 1, 1] = complex(row[5], row[6])
            S_true[i, 0, 1] = complex(row[7], row[8])

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
            mag = 10**(pred_db[p_idx, :]/20.0)
            S_corrected[:, p, q] = mag * np.exp(1j*0)

        def to_db(S):
            db = np.zeros((len(freqs), 4), dtype=float)
            for i, (p, q) in enumerate([(0,0),(1,0),(1,1),(0,1)]):
                db[:, i] = 20*np.log10(np.abs(S[:, p, q]) + 1e-12)
            return db

        db_true = to_db(S_true)
        db_corrected = to_db(S_corrected)

        print(f"\nSample {n_target} -- per-band MAE (dB):")
        bands = [(0.5, 4.5), (4.0, 8.5), (8.0, 12.5), (12.0, 15.0)]
        names = ["S11", "S21", "S22", "S12"]
        for name, idx in zip(names, range(4)):
            parts = []
            for low, high in bands:
                mask_band = (freqs >= low) & (freqs <= high)
                if np.any(mask_band):
                    mae = np.mean(np.abs(db_corrected[mask_band, idx] - db_true[mask_band, idx]))
                    parts.append(f"{low}-{high}: {mae:.2f}")
            print(f"  {name}: {' | '.join(parts)}")

        if is_range:
            pdf_path = output_dir / f"predict_n{n_target}.pdf"
        else:
            if args.output_pdf:
                pdf_path = Path(args.output_pdf)
            else:
                pdf_path = output_dir / f"predict_n{n_target}.pdf"

        seq_str = ' -> '.join([f"{t}:{n}" for t, n in seq_info])
        plot_comparison(freqs, db_true, db_corrected, seq_str, pdf_path)

    print("All predictions complete!")


if __name__ == "__main__":
    main()
