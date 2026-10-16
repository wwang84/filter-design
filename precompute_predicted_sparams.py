#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_predicted_sparams.py

Runs inference with subdevice surrogate models for all unique (family, n) devices
and caches predicted S-parameters (8 channels RI).

Purpose:
  During cascade model training, use subdevice surrogate model predictions
  instead of HFSS ground truth to eliminate train/inference distribution shift.

Usage:
    python precompute_predicted_sparams.py

Output:
    ./predicted_sparams_cache.npz  - predicted S-params
    ./predicted_sparams_cache.json - metadata
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from model_block import load_trained_model, predict_sparams

# ============================================================
# Path configuration
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
CKPT_DIR = ROOT_DIR / "checkpoints"
CACHE_NPZ = ROOT_DIR / "predicted_sparams_cache.npz"
CACHE_JSON = ROOT_DIR / "predicted_sparams_cache.json"

FAMILIES = ["single_step", "double_step", "open_stub", "t_stub", "tapered_line"]

FREQ_START = 0.5
FREQ_END = 15.0
FREQ_STEP = 0.1
N_FREQ = int((FREQ_END - FREQ_START) / FREQ_STEP) + 1  # 146


def get_unique_devices():
    """Extract all unique (family, n) from the cascade dataset."""
    with open(DATASET_DIR / "cascade" / "train.json") as f:
        cascade_data = json.load(f)

    unique = set()
    for item in cascade_data:
        for fam, n in item["sequence"]:
            unique.add((fam, n))

    return sorted(unique)  # sort for reproducibility


def load_geometry(family: str, n: int):
    """Load geometry parameters for a specific family + n from the subdevice dataset."""
    train_json = DATASET_DIR / family / "train.json"
    with open(train_json) as f:
        data = json.load(f)

    for item in data:
        if item["n"] == n:
            x_order = item["x_order"]
            geom = {}
            for name, val in zip(x_order, item["x"][0]):
                if name != "f_ghz":
                    geom[name] = float(val)
            return x_order, geom

    return None, None


def build_input_tensor(x_order, geom):
    """Build (N_FREQ, input_dim) input tensor."""
    freqs = np.linspace(FREQ_START, FREQ_END, N_FREQ)
    rows = []
    for f in freqs:
        row = []
        for name in x_order:
            if name == "f_ghz":
                row.append(f)
            else:
                row.append(geom[name])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frequency range: {FREQ_START}-{FREQ_END} GHz, {N_FREQ} points")

    # Get all unique devices
    unique_devices = get_unique_devices()
    print(f"\nUnique (family, n) combinations: {len(unique_devices)}")
    by_family = defaultdict(list)
    for fam, n in unique_devices:
        by_family[fam].append(n)
    for fam in FAMILIES:
        print(f"  {fam}: {len(by_family.get(fam, []))}")

    # Run inference and cache
    cache = {}

    for family in FAMILIES:
        model_path = CKPT_DIR / family / "model.pt"
        if not model_path.exists():
            print(f"\nWarning: {family} model not found at {model_path}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Loading {family} model: {model_path}")
        model, norm_stats, cfg = load_trained_model(model_path, device=device)
        x_order = cfg["x_order"]
        print(f"  x_order: {x_order}")
        print(f"  params: hidden={cfg.get('hidden')}, depth={cfg.get('depth')}, "
              f"fourier_bands={cfg.get('fourier_bands')}")

        family_devices = [(family, n) for n in by_family.get(family, [])]
        print(f"  devices to infer: {len(family_devices)}")

        for fam, n in sorted(family_devices, key=lambda x: x[1]):
            x_order_dev, geom = load_geometry(family, n)
            if geom is None:
                print(f"    Warning: {family} n={n} not found in dataset")
                continue

            x_tensor = build_input_tensor(x_order_dev, geom)
            # y_pred: (N_FREQ, 8) - [s11_re, s11_im, s21_re, s21_im, s22_re, s22_im, s12_re, s12_im]
            y_pred = predict_sparams(model, norm_stats, x_tensor, device=device)

            key = f"{family}_{n}"
            cache[key] = y_pred.numpy().astype(np.float32)

            if len(cache) % 50 == 0:
                print(f"    progress: {len(cache)}/{len(unique_devices)}")

        # Free GPU memory
        del model

    # ---- Save ----
    print(f"\n{'=' * 60}")
    print(f"Saving cache: {CACHE_NPZ}")
    np.savez_compressed(CACHE_NPZ, **cache)
    print(f"  Cached {len(cache)} devices")

    # Metadata
    meta = {
        "keys": sorted(cache.keys()),
        "n_freq": N_FREQ,
        "freq_start": FREQ_START,
        "freq_end": FREQ_END,
        "freq_step": FREQ_STEP,
        "description": "Subdevice surrogate model predicted S-parameters (8 channels RI)",
    }
    with open(CACHE_JSON, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata: {CACHE_JSON}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
