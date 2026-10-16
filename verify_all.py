#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_all.py

Comprehensive validation script for the entire cascade correction project.
Run this on the server to verify every component works correctly.
Auto-detects GPU and uses it when available.
Prints PASS/FAIL for each step.

IMPORTANT: tests run in order and share cached data to avoid repeated expensive loads.
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
SKIP = 0

# Shared cache between tests (loaded once)
CACHE = {"sub_map": None, "cascade": None}


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(name: str):
    def decorator(func):
        def wrapper():
            global PASS, FAIL, SKIP
            try:
                func()
                print(f"  ✓ PASS: {name}")
                PASS += 1
            except SkipTest as e:
                print(f"  ○ SKIP: {name} — {e}")
                SKIP += 1
            except Exception as e:
                print(f"  ✗ FAIL: {name}")
                print(f"    Error: {e}")
                traceback.print_exc(file=sys.stdout)
                FAIL += 1
        return wrapper
    return decorator


class SkipTest(Exception):
    pass


# ============================================================
# Shared data loading (called once, then cached)
# ============================================================

def load_sub_map():
    if CACHE["sub_map"] is None:
        from dataset import load_all_subdevices
        CACHE["sub_map"] = load_all_subdevices(ROOT / "dataset")
    return CACHE["sub_map"]


def load_cascade():
    if CACHE["cascade"] is None:
        from dataset import load_cascade_data
        CACHE["cascade"] = load_cascade_data(ROOT / "dataset/cascade", load_sub_map())
    return CACHE["cascade"]


# ============================================================
# Tests
# ============================================================

@check("All imports (model, loss, dataset, utils)")
def test_imports():
    from model import ResidualNet
    from loss import TotalLoss, ReciprocalLoss, SelectiveSmoothnessLoss
    from dataset import (
        ResidualDataset, collate_fn, load_all_subdevices,
        load_cascade_data, load_predicted_sparams_cache,
        build_dataloaders_by_n_split, compute_cascade,
        flatten_device_s_to_tensor, _compute_adjacent_width_diffs
    )
    from utils import (
        TrainLogger, set_seed, count_parameters, ensure_dir,
        format_time, save_model_with_stats, get_device
    )
    assert ResidualNet.__name__ == "ResidualNet"
    assert TotalLoss.__name__ == "TotalLoss"
    assert ResidualDataset.__name__ == "ResidualDataset"


@check("Model: instantiation + forward + backward")
def test_model():
    import torch
    from model import ResidualNet

    device = get_device()
    B, L, F = 4, 6, 146
    model = ResidualNet(n_freq=F, max_len=L).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_params:,}  Device: {device}")

    devices_s = torch.randn(B, L, 9, F).to(device)
    mask = torch.zeros(B, L, dtype=torch.bool).to(device)
    mask[0, :2] = mask[1, :3] = mask[2, :4] = mask[3, :6] = True
    freq_norm = torch.rand(B, F).to(device)

    pred = model(devices_s, mask, freq_norm)
    assert pred.shape == (B, 4 * F), f"Expected (B, 4*F), got {pred.shape}"

    loss = pred.mean()
    loss.backward()
    for _name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            break
    else:
        raise AssertionError("No parameter received gradients")


@check("Loss: TotalLoss forward")
def test_loss():
    import torch
    from loss import TotalLoss

    device = get_device()
    B, F = 4, 146
    pred = torch.randn(B, 4 * F).to(device)
    target = torch.randn(B, 4 * F).to(device)

    loss_fn = TotalLoss()
    loss, db_mae, *_ = loss_fn(pred, target, F, None, None)
    assert loss.shape == ()
    assert db_mae >= 0

    num_devices = torch.ones(B).to(device)
    loss2, *_ = loss_fn(pred, target, F, num_devices, None)
    assert torch.isfinite(loss2)


@check("Data: subdevice loading (without measured/)")
def test_data_subdevice_loading():
    sub_map = load_sub_map()
    families = sorted(set(k[0] for k in sub_map.keys()))
    print(f"      Families: {families}")
    print(f"      Total devices: {len(sub_map)}")
    assert "measured" not in families, "measured/ dir was included"
    expected = {"single_step", "double_step", "open_stub", "t_stub", "tapered_line"}
    assert set(families) == expected


@check("Data: cascade loading")
def test_data_cascade_loading():
    cascade = load_cascade()
    print(f"      Cascade samples: {len(cascade)}")
    s = cascade[0]
    required_keys = {"freqs", "S_cascade", "S_subs", "S_true", "geoms", "seq_info", "port_widths"}
    assert required_keys.issubset(s.keys())


@check("Data: cascade with predicted S-param cache")
def test_data_with_predicted_cache():
    from dataset import load_predicted_sparams_cache

    cache_path = ROOT / "predicted_sparams_cache.npz"
    if not cache_path.exists():
        raise SkipTest("predicted_sparams_cache.npz not found (run precompute_predicted_sparams.py first)")

    cache = load_predicted_sparams_cache(cache_path)
    print(f"      Cache entries: {len(cache)}")
    assert len(cache) > 0

    sub_map = load_sub_map()
    from dataset import load_cascade_data
    cascade = load_cascade_data(ROOT / "dataset/cascade", sub_map, predicted_sparams_cache=cache)
    assert len(cascade) > 0


@check("Data: DataLoader + batch iteration")
def test_dataloader():
    # Use only the first 1000 samples for speed
    from dataset import build_dataloaders_by_n_split
    cascade = load_cascade()
    small = cascade[:1000]
    train_loader, _, train_ds, val_ds, stats = build_dataloaders_by_n_split(
        small, output_mode="db_phase", batch_size=16, max_len=6,
        train_ratio=0.8, random_state=42
    )
    batch = next(iter(train_loader))
    devices_s, mask, freq_norm, target = batch
    assert mask.shape[-1] == 6
    assert target.shape[-1] == 4 * train_ds.n_freq
    print(f"      n_freq={train_ds.n_freq}, train={len(train_ds)}, val={len(val_ds)}")


@check("Data: augmentation + noise")
def test_augmentation():
    from dataset import build_dataloaders_by_n_split
    cascade = load_cascade()
    train_loader, _, _, _, _ = build_dataloaders_by_n_split(
        cascade[:100], output_mode="db_phase", batch_size=4, max_len=6,
        train_ratio=0.8, random_state=42, freq_range=(0.5, 4.5)
    )
    batch = next(iter(train_loader))
    assert len(batch) == 4


@check("Model → Loss → Backward (end-to-end mini pipeline)")
def test_end_to_end():
    import torch
    from dataset import build_dataloaders_by_n_split
    from model import ResidualNet
    from loss import TotalLoss

    device = get_device()
    cascade = load_cascade()
    train_loader, _, train_ds, _, _ = build_dataloaders_by_n_split(
        cascade[:1000], output_mode="db_phase", batch_size=8, max_len=6,
        train_ratio=0.8, random_state=42
    )

    model = ResidualNet(n_freq=train_ds.n_freq, max_len=6).to(device)
    loss_fn = TotalLoss()

    devices_s, mask, freq_norm, target = next(iter(train_loader))
    devices_s, mask, freq_norm, target = [t.to(device) for t in (devices_s, mask, freq_norm, target)]
    pred = model(devices_s, mask, freq_norm)
    loss, db_mae, *_ = loss_fn(pred, target, train_ds.n_freq, None, None)
    loss.backward()

    assert torch.isfinite(loss)
    print(f"      loss={loss.item():.4f}, db_mae={db_mae.item():.4f}  Device: {device}")


@check("Training: 1 step (verify optimizer + backward)")
def test_train_one_step():
    import torch
    import torch.optim as optim
    from dataset import build_dataloaders_by_n_split
    from model import ResidualNet
    from loss import TotalLoss

    device = get_device()
    cascade = load_cascade()
    train_loader, _, train_ds, _, _ = build_dataloaders_by_n_split(
        cascade[:1000], output_mode="db_phase", batch_size=8, max_len=6,
        train_ratio=0.8, random_state=42, freq_range=(0.5, 4.5)
    )

    model = ResidualNet(n_freq=train_ds.n_freq, max_len=6).to(device)
    loss_fn = TotalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    devices_s, mask, freq_norm, target = next(iter(train_loader))
    devices_s, mask, freq_norm, target = [t.to(device) for t in (devices_s, mask, freq_norm, target)]

    optimizer.zero_grad()
    pred = model(devices_s, mask, freq_norm)
    loss, *_ = loss_fn(pred, target, train_ds.n_freq, None, None)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(loss)

    assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss}"
    print(f"      loss={loss.item():.4f}  Device: {device}")


@check("Precompute: can import model_block functions")
def test_precompute_imports():
    from model_block import load_trained_model, predict_sparams
    print(f"      OK: load_trained_model, predict_sparams")


@check("Precompute: dataset geometry structure")
def test_precompute_geometry():
    import json
    for family in ["single_step", "double_step", "open_stub", "t_stub", "tapered_line"]:
        f = ROOT / "dataset" / family / "train.json"
        with open(f) as fh:
            data = json.load(fh)
        assert len(data) > 0, f"Empty: {family}"
        item = data[0]
        for k in ["n", "x", "y", "x_order"]:
            assert k in item, f"{family}: missing '{k}'"
        print(f"      {family}: {len(data)} samples OK")


@check("Precompute: checkpoint files exist")
def test_checkpoints_exist():
    for family in ["single_step", "double_step", "open_stub", "t_stub", "tapered_line"]:
        model_path = ROOT / "checkpoints" / family / "model.pt"
        config_path = ROOT / "checkpoints" / family / "config.json"
        norm_path = ROOT / "checkpoints" / family / "norm_stats.pt"
        missing = [
            p.name for p, name in [(model_path, "model.pt"), (config_path, "config.json"), (norm_path, "norm_stats.pt")]
            if not p.exists()
        ]
        if missing:
            raise SkipTest(f"{family} missing: {', '.join(missing)}")
        print(f"      {family}: all checkpoint files present")


@check("GPU availability")
def test_gpu():
    import torch
    if torch.cuda.is_available():
        print(f"      CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"      CPU only")


@check("Predict: import + function signatures")
def test_predict_imports():
    from predict import (
        build_prediction_input, denormalize_prediction,
        plot_comparison, parse_n_range
    )
    assert callable(parse_n_range)
    rng = parse_n_range("10-20")
    assert rng == (True, 10, 20)
    single = parse_n_range("5")
    assert single == (False, 5, None)
    print(f"      All functions importable")


@check("Test: import + function signatures")
def test_test_imports():
    from test import (
        build_prediction_input, denormalize_prediction,
        compute_detailed_errors, load_model_auto
    )
    assert callable(compute_detailed_errors)
    assert callable(load_model_auto)
    print(f"      All functions importable")


# ============================================================
# Main
# ============================================================

def main():
    global PASS, FAIL, SKIP

    print("=" * 70)
    print("  Cascade Correction Project — Comprehensive Verification")
    print("=" * 70)
    print()

    # ── First: import / model / loss / small-data tests (no cascade load needed) ──
    print("[Imports]")
    test_imports()
    print()

    print("[Model/Loss]")
    test_model()
    test_loss()
    print()

    # ── Now load the full cascade data once, then run all data-dependent tests ──
    print("[Data Pipeline — loading cascade data (this may take a moment)]")
    # Trigger caching: sub_map then cascade
    load_sub_map()
    load_cascade()

    # Data tests — reuse cached cascade
    test_data_subdevice_loading()
    test_data_cascade_loading()
    test_data_with_predicted_cache()
    test_dataloader()
    test_augmentation()
    print()

    print("[End-to-End]")
    test_end_to_end()
    test_train_one_step()
    print()

    print("[Precompute]")
    test_precompute_imports()
    test_precompute_geometry()
    test_checkpoints_exist()
    print()

    print("[Other Modules]")
    test_gpu()
    test_predict_imports()
    test_test_imports()
    print()

    # ── Summary ──
    total = PASS + FAIL + SKIP
    print("=" * 70)
    print(f"  Results: {PASS} PASS / {FAIL} FAIL / {SKIP} SKIP (total {total})")
    print("=" * 70)

    if FAIL > 0:
        print("\n⚠  Some checks FAILED. Review the output above.")
        sys.exit(1)
    else:
        print("\n✓ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
