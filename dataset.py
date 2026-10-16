#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py - Pure S-parameter version (no type, no n)
- Input: device S-parameters only (8 channels RI)
- Target: S_true dB (4 channels)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random

GEOM_ORDER = {
    "single_step":  ["Lsub", "Ls", "Ws", "dummy", "dummy", "dummy", "dummy"],
    "double_step":  ["Lsub", "L1", "W1", "L2", "W2", "dummy", "dummy"],
    "open_stub":    ["Lsub", "Lmain", "x_stub", "Lstub", "Wstub", "dummy", "dummy"],
    "t_stub":       ["Lsub", "Lmain", "x_stub", "Lstem", "Wstem", "Lcap", "Wcap"],
    "tapered_line": ["Lsub", "L_mid", "P_left", "P_right", "W_left", "W_right", "dummy"],
}


def get_port_widths(family, geom):
    if family == "tapered_line":
        if isinstance(geom, np.ndarray):
            return float(geom[3]), float(geom[4])
        left = geom.get("P_left", geom.get("W_left", 1.8))
        right = geom.get("P_right", geom.get("W_right", 1.8))
        return float(left), float(right)
    return 1.8, 1.8


def load_predicted_sparams_cache(cache_path: Path):
    """Load precomputed predicted S-parameter cache (.npz file)"""
    data = np.load(cache_path)
    cache = {}
    for key in data.keys():
        # key format: "family_n" -> S array (n_freq, 8)
        cache[key] = data[key].copy()  # load immediately to memory
    data.close()
    return cache


def _compute_adjacent_width_diffs(seq, geoms):
    """Adjacent device port width differences (normalized to [0,1]). Normalization factor = 5.0mm."""
    diffs = [0.0]
    for i in range(1, len(seq)):
        prev_fam = seq[i-1][0]
        curr_fam = seq[i][0]
        _, prev_right = get_port_widths(prev_fam, geoms[i-1])
        curr_left, _ = get_port_widths(curr_fam, geoms[i])
        diffs.append(float(abs(curr_left - prev_right) / 5.0))
    return diffs


def extract_geom_vector(x_order, x_values):
    vec = np.zeros(7, dtype=np.float32)
    geom_dict = {name: val for name, val in zip(x_order[1:], x_values)}
    for i, name in enumerate(GEOM_ORDER["t_stub"]):
        if name in geom_dict:
            vec[i] = geom_dict[name]
    return vec


def s2t(S, eps=1e-12):
    S11, S12, S21, S22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    S21_safe = S21 if np.abs(S21) >= eps else eps * np.exp(1j * np.angle(S21))
    T11 = -(S11 * S22 - S12 * S21) / S21_safe
    T12 = S11 / S21_safe; T21 = -S22 / S21_safe; T22 = 1.0 / S21_safe
    return np.array([[T11, T12], [T21, T22]])


def t2s(T, eps=1e-12):
    T11, T12, T21, T22 = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
    T22_safe = T22 if np.abs(T22) >= eps else eps * np.exp(1j * np.angle(T22))
    S11 = T12 / T22_safe; S12 = (T11 * T22_safe - T12 * T21) / T22_safe
    S21 = 1.0 / T22_safe; S22 = -T21 / T22_safe
    return np.array([[S11, S12], [S21, S22]])


def cascade_S(S_list):
    T = s2t(S_list[0])
    for S in S_list[1:]:
        T = T @ s2t(S)
    return t2s(T)


def compute_cascade(S_subs):
    n_freq = S_subs[0].shape[0]
    S_cascade = np.zeros((n_freq, 2, 2), dtype=complex)
    for i in range(n_freq):
        S_cascade[i] = cascade_S([S_sub[i] for S_sub in S_subs])
    return S_cascade


def load_subdevice_data(subdevice_dir):
    with open(subdevice_dir / "train.json") as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        n = item['n']
        x = np.array(item['x']); y = np.array(item['y'])
        freqs = x[:, 0]; geom = extract_geom_vector(item['x_order'], x[0, 1:])
        S = np.zeros((len(freqs), 2, 2), dtype=complex)
        S[:, 0, 0] = y[:, 0] + 1j * y[:, 1]; S[:, 1, 0] = y[:, 2] + 1j * y[:, 3]
        S[:, 1, 1] = y[:, 4] + 1j * y[:, 5]; S[:, 0, 1] = y[:, 6] + 1j * y[:, 7]
        mapping[(subdevice_dir.name, n)] = {"S": S, "geom": geom, "freqs": freqs}
    return mapping


FAMILIES = ["single_step", "double_step", "open_stub", "t_stub", "tapered_line"]


def load_all_subdevices(root_dir):
    all_maps = {}
    for family in FAMILIES:
        sub_dir = root_dir / family
        if sub_dir.is_dir():
            all_maps.update(load_subdevice_data(sub_dir))
    return all_maps


def _predicted_s_to_ri(predicted_y, n_freq):
    """Convert predicted S-parameters (N_FREQ, 8) to complex S matrix"""
    S = np.zeros((n_freq, 2, 2), dtype=complex)
    S[:, 0, 0] = predicted_y[:, 0] + 1j * predicted_y[:, 1]
    S[:, 1, 0] = predicted_y[:, 2] + 1j * predicted_y[:, 3]
    S[:, 1, 1] = predicted_y[:, 4] + 1j * predicted_y[:, 5]
    S[:, 0, 1] = predicted_y[:, 6] + 1j * predicted_y[:, 7]
    return S


def load_cascade_data(cascade_dir, subdevice_map, predicted_sparams_cache=None):
    with open(cascade_dir / "train.json") as f:
        data = json.load(f)
    samples = []
    n_pred_used = 0
    for item in data:
        seq = item['sequence']
        S_subs = []
        for d, n in seq:
            if predicted_sparams_cache is not None:
                key = f"{d}_{n}"
                if key in predicted_sparams_cache:
                    S_subs.append(predicted_sparams_cache[key])
                    n_pred_used += 1
                    continue
            S_subs.append(subdevice_map[(d, n)]["S"])

        n_freq_sub = S_subs[0].shape[0]
        S_subs_complex = []
        for s in S_subs:
            if s.ndim == 2 and s.shape[1] == 8:
                S_subs_complex.append(_predicted_s_to_ri(s, n_freq_sub))
            else:
                S_subs_complex.append(s)

        geoms = [subdevice_map[(d, n)]["geom"] for d, n in seq]
        y_true = np.array(item['s'])
        freqs = y_true[:, 0]
        S_true = np.zeros((y_true.shape[0], 2, 2), dtype=complex)
        S_true[:, 0, 0] = y_true[:, 1] + 1j * y_true[:, 2]
        S_true[:, 1, 0] = y_true[:, 3] + 1j * y_true[:, 4]
        S_true[:, 1, 1] = y_true[:, 5] + 1j * y_true[:, 6]
        S_true[:, 0, 1] = y_true[:, 7] + 1j * y_true[:, 8]
        S_cascade = compute_cascade(S_subs_complex)
        port_widths = _compute_adjacent_width_diffs(seq, geoms)
        samples.append({'freqs': freqs, 'S_cascade': S_cascade, 'S_subs': S_subs_complex,
                         'S_true': S_true, 'geoms': geoms, 'seq_info': seq,
                         'port_widths': port_widths})

    if predicted_sparams_cache is not None and n_pred_used > 0:
        print(f"  Using predicted S-params: {n_pred_used} subdevice instances")
    return samples


def flatten_device_s_to_tensor(S_device, n_freq):
    out = np.zeros((8, n_freq), dtype=np.float32)
    out[0, :] = S_device[:, 0, 0].real; out[1, :] = S_device[:, 0, 0].imag
    out[2, :] = S_device[:, 1, 0].real; out[3, :] = S_device[:, 1, 0].imag
    out[4, :] = S_device[:, 1, 1].real; out[5, :] = S_device[:, 1, 1].imag
    out[6, :] = S_device[:, 0, 1].real; out[7, :] = S_device[:, 0, 1].imag
    return out


def get_freq_indices_in_range(freqs, freq_range):
    if freq_range is None: return np.arange(len(freqs))
    low, high = freq_range; mask = (freqs >= low) & (freqs <= high)
    return np.where(mask)[0]


def compute_target_freq_std(samples):
    if len(samples) == 0: return None
    n_freq = len(samples[0]['freqs'])
    all_db = np.zeros((len(samples), n_freq), dtype=np.float32)
    for i, sample in enumerate(samples):
        S_true = sample['S_true']
        mag = np.abs(S_true[:, 1, 0]) + 1e-12
        all_db[i] = 20 * np.log10(mag)
    freq_std = np.std(all_db, axis=0)
    return freq_std


def compute_freq_errors(samples):
    total_samples = len(samples)
    if total_samples == 0: return np.array([]), np.array([])
    n_freq = len(samples[0]['freqs'])
    freq_errors = np.zeros(n_freq, dtype=np.float32); freq_counts = np.zeros(n_freq, dtype=np.int32)
    for sample in samples:
        S_cascade = sample['S_cascade']; S_true = sample['S_true']
        for p, q in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            mag_cas = np.abs(S_cascade[:, p, q]) + 1e-12; mag_true = np.abs(S_true[:, p, q]) + 1e-12
            db_cas = 20 * np.log10(mag_cas); db_true = 20 * np.log10(mag_true)
            err = np.abs(db_cas - db_true); freq_errors += err; freq_counts += 1
    freq_errors = freq_errors / (freq_counts + 1e-8)
    sorted_indices = np.argsort(freq_errors)[::-1]; sorted_errors = freq_errors[sorted_indices]
    return sorted_indices, sorted_errors


def select_hard_frequencies(samples, n_selected=30):
    if n_selected <= 0: return None
    sorted_indices, _ = compute_freq_errors(samples)
    n_freq = len(sorted_indices); n_selected = min(n_selected, n_freq)
    if n_selected == n_freq: return None
    selected_indices = np.sort(sorted_indices[:n_selected])
    print(f"Selected {n_selected} hard frequencies (out of {n_freq})"); return selected_indices


def split_by_n_values(samples, train_ratio=0.8, random_state=42):
    perm_groups = defaultdict(list)
    for idx, s in enumerate(samples):
        perm = tuple(fam for fam, _ in s['seq_info'])
        perm_groups[perm].append(idx)
    train_idx, val_idx = [], []
    rng = random.Random(random_state)
    for perm, indices in perm_groups.items():
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_ratio))
        train_idx.extend(indices[:n_train]); val_idx.extend(indices[n_train:])
    return train_idx, val_idx


# ============================================================
# Collate: S-params + mask + freq only, no type/n
# ============================================================
def collate_fn(batch):
    max_len = max(item[0].shape[0] for item in batch)
    n_freq = batch[0][0].shape[-1]
    n_ch = batch[0][0].shape[1]
    target_dim = batch[0][-1].shape[0]
    B = len(batch)
    devices_s = torch.zeros(B, max_len, n_ch, n_freq, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    freq_norm = torch.zeros(B, n_freq, dtype=torch.float32)
    targets = torch.zeros(B, target_dim, dtype=torch.float32)
    for i, (d_s, m, fn, tg) in enumerate(batch):
        L = d_s.shape[0]
        devices_s[i, :L] = d_s
        mask[i, :L] = m
        freq_norm[i] = fn
        targets[i] = tg
    return devices_s, mask, freq_norm, targets


# ============================================================
# Dataset: S-params -> S_true (no type, no n)
# ============================================================
class ResidualDataset(Dataset):
    """S-params -> S_true (no type, no n)"""

    def __init__(self, samples, output_mode='db_phase',
                 freq_min=None, freq_max=None, max_len=6,
                 augment=False, noise_std=0.01,
                 target_mean=None, target_std=None,
                 hard_freq_indices=None, freq_range=None):
        self.samples = samples
        self.output_mode = output_mode; self.max_len = max_len
        self.augment = augment; self.noise_std = noise_std; self.freq_range = freq_range

        if len(samples) == 0:
            self.hard_freq_indices = np.array([]); self.n_freq = 0; self.original_n_freq = 0; return

        all_freqs = samples[0]['freqs']; self.original_n_freq = len(all_freqs)
        if freq_range is not None:
            base_indices = get_freq_indices_in_range(all_freqs, freq_range)
            print(f"Band {freq_range[0]:.1f}-{freq_range[1]:.1f} GHz: {len(base_indices)} frequency points")
        else:
            base_indices = np.arange(self.original_n_freq)

        if hard_freq_indices is not None:
            hard_set = set(hard_freq_indices); base_set = set(base_indices)
            final_indices = np.sort(list(base_set & hard_set))
            if len(final_indices) == 0: final_indices = base_indices
        else:
            final_indices = base_indices

        self.hard_freq_indices = final_indices; self.n_selected = len(self.hard_freq_indices)
        self.n_freq = self.n_selected
        print(f"Actual frequency points used: {self.n_selected} (original {self.original_n_freq})")

        if freq_min is None or freq_max is None:
            if len(self.hard_freq_indices) > 0:
                freqs_used = all_freqs[self.hard_freq_indices]
                self.freq_min = np.min(freqs_used); self.freq_max = np.max(freqs_used)
            else:
                self.freq_min = 0.5; self.freq_max = 15.0
        else:
            self.freq_min = freq_min; self.freq_max = freq_max

        # ----- Subdevice S-parameter normalization (8 channels) + reciprocity (S21~S12) + port width difference -----
        all_device_s_by_ch = {i: [] for i in range(9)}
        for sample in samples:
            for i, S_sub in enumerate(sample['S_subs']):
                S_sub_sel = S_sub[self.hard_freq_indices]
                for ch in range(8):
                    if ch % 2 == 0:
                        vals = S_sub_sel[:, ch//2//2, ch//2%2].real
                    else:
                        vals = S_sub_sel[:, ch//2//2, ch//2%2].imag
                    all_device_s_by_ch[ch].extend(vals.tolist())
                # Channel 9: adjacent port width difference (frequency-independent)
                if 'port_widths' in sample and i < len(sample['port_widths']):
                    all_device_s_by_ch[8].append(sample['port_widths'][i])
        self.device_s_mean = np.zeros(9, dtype=np.float32); self.device_s_std = np.ones(9, dtype=np.float32)
        for ch in range(9):
            arr = np.array(all_device_s_by_ch[ch])
            self.device_s_mean[ch] = np.mean(arr); self.device_s_std[ch] = np.std(arr) + 1e-8

        # ----- Target: S_true dB (4 channels) -----
        all_targets_by_channel = {i: [] for i in range(4)}
        for sample in samples:
            S_true = sample['S_true'][self.hard_freq_indices]
            ports = [(0,0),(1,0),(1,1),(0,1)]
            target = np.zeros(4*self.n_selected, dtype=np.float32)
            for p_idx, (p,q) in enumerate(ports):
                mag_true = np.abs(S_true[:,p,q]) + 1e-12
                db_true = 20*np.log10(mag_true)
                target[p_idx*self.n_selected:(p_idx+1)*self.n_selected] = db_true
            target_reshaped = target.reshape(4, self.n_selected)
            for ch in range(4):
                all_targets_by_channel[ch].extend(target_reshaped[ch].tolist())

        if target_mean is not None and target_std is not None:
            self.target_mean_per_channel = target_mean; self.target_std_per_channel = target_std
        else:
            self.target_mean_per_channel = np.zeros(4, dtype=np.float32)
            self.target_std_per_channel = np.ones(4, dtype=np.float32)
            for ch in range(4):
                arr = np.array(all_targets_by_channel[ch])
                self.target_mean_per_channel[ch] = np.mean(arr)
                self.target_std_per_channel[ch] = np.std(arr) + 1e-8

        # ----- Precompute cache -----
        self.cache = []
        for sample in samples:
            S_true = sample['S_true'][self.hard_freq_indices]
            freqs_selected = sample['freqs'][self.hard_freq_indices]
            seq_info = sample['seq_info']; num_dev = len(seq_info)

            # Target normalization
            ports = [(0,0),(1,0),(1,1),(0,1)]
            target = np.zeros(4*self.n_selected, dtype=np.float32)
            for p_idx, (p,q) in enumerate(ports):
                mag_true = np.abs(S_true[:,p,q]) + 1e-12
                db_true = 20*np.log10(mag_true)
                target[p_idx*self.n_selected:(p_idx+1)*self.n_selected] = db_true
            target_reshaped = target.reshape(4, self.n_selected)
            normalized_target = np.zeros((4, self.n_selected), dtype=np.float32)
            for ch in range(4):
                m = self.target_mean_per_channel[ch]; s = self.target_std_per_channel[ch]
                normalized_target[ch] = (target_reshaped[ch] - m) / s
            target_norm_flat = normalized_target.reshape(-1)

            freq_norm = (freqs_selected - self.freq_min) / (self.freq_max - self.freq_min + 1e-8)
            freq_norm = freq_norm.astype(np.float32)

            # Subdevice S-params (9 channels: 8ch RI + 1ch port_width_diff)
            devices_s = np.zeros((num_dev, 9, self.n_selected), dtype=np.float32)
            for i, S_sub in enumerate(sample['S_subs']):
                if i >= num_dev: break
                s_tensor_8ch = flatten_device_s_to_tensor(S_sub[self.hard_freq_indices], self.n_selected)
                s_tensor = np.zeros((9, self.n_selected), dtype=np.float32)
                for ch in range(8):
                    s_tensor[ch] = (s_tensor_8ch[ch] - self.device_s_mean[ch]) / self.device_s_std[ch]
                # Channel 9: adjacent port width difference (same value broadcast to all frequency points)
                pw_val = 0.0
                if 'port_widths' in sample and i < len(sample['port_widths']):
                    pw_val = float(sample['port_widths'][i])
                s_tensor[8] = ((pw_val - self.device_s_mean[8]) / self.device_s_std[8])
                devices_s[i] = s_tensor

            mask = np.ones(num_dev, dtype=bool)

            self.cache.append({
                'devices_s': devices_s,
                'mask': mask,
                'freq_norm': freq_norm,
                'target': target_norm_flat,
            })

    def __len__(self): return len(self.cache)

    def __getitem__(self, idx):
        item = self.cache[idx]
        devices_s = torch.from_numpy(item['devices_s'])
        mask = torch.from_numpy(item['mask'])
        freq_norm = torch.from_numpy(item['freq_norm'])
        target = torch.from_numpy(item['target'])

        if self.augment:
            d_s_np = devices_s.numpy().copy()
            for ch in range(0, 8, 2):
                real = d_s_np[:, ch, :]; imag = d_s_np[:, ch+1, :]
                amp = np.sqrt(real**2 + imag**2 + 1e-12)
                noise = np.random.normal(0, self.noise_std*amp, size=amp.shape).astype(np.float32)
                d_s_np[:, ch, :] += noise; d_s_np[:, ch+1, :] += noise
            devices_s = torch.from_numpy(d_s_np)

        return (devices_s, mask, freq_norm, target)


# ============================================================
# DataLoader construction
# ============================================================
def build_dataloaders_by_n_split(samples, output_mode='db_phase',
    batch_size=32, num_workers=0, augment=True, noise_std=0.02,
    freq_min=None, freq_max=None, max_len=6, train_ratio=0.8,
    random_state=42, n_selected=0, freq_range=None):
    """DataLoader split by n value"""
    train_idx, val_idx = split_by_n_values(samples, train_ratio, random_state)
    train_samples = [samples[i] for i in train_idx]; val_samples = [samples[i] for i in val_idx]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)} (split by n value)")

    hard_freq_indices = None if freq_range is not None else select_hard_frequencies(train_samples, n_selected)

    train_ds = ResidualDataset(train_samples, output_mode=output_mode,
        freq_min=freq_min, freq_max=freq_max, max_len=max_len,
        augment=augment, noise_std=noise_std,
        hard_freq_indices=hard_freq_indices, freq_range=freq_range)

    freq_min = train_ds.freq_min; freq_max = train_ds.freq_max
    target_mean = train_ds.target_mean_per_channel; target_std = train_ds.target_std_per_channel
    device_s_mean = train_ds.device_s_mean; device_s_std = train_ds.device_s_std

    freq_std_target = compute_target_freq_std(train_samples)
    if freq_std_target is not None and hard_freq_indices is not None:
        freq_std_target = freq_std_target[hard_freq_indices]

    val_ds = ResidualDataset(val_samples, output_mode=output_mode,
        freq_min=freq_min, freq_max=freq_max, max_len=max_len, augment=False,
        target_mean=target_mean, target_std=target_std,
        hard_freq_indices=hard_freq_indices, freq_range=freq_range)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers>0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers>0, collate_fn=collate_fn)

    stats = {
        'freq_min': freq_min, 'freq_max': freq_max,
        'target_mean_per_channel': target_mean, 'target_std_per_channel': target_std,
        'device_s_mean': device_s_mean, 'device_s_std': device_s_std,
        'max_len': max_len,
        'freq_std_target': freq_std_target.tolist() if freq_std_target is not None else None,
    }
    return train_loader, val_loader, train_ds, val_ds, stats


# Backward compatibility
def build_dataloaders(train_samples, val_samples, **kwargs):
    hard_freq_indices = None if kwargs.get('freq_range') else select_hard_frequencies(train_samples, kwargs.get('n_selected', 0))
    train_ds = ResidualDataset(train_samples,
        **{k:v for k,v in kwargs.items() if k in ['output_mode','freq_min','freq_max','max_len','augment','noise_std','hard_freq_indices','freq_range']},
        hard_freq_indices=hard_freq_indices)
    val_ds = ResidualDataset(val_samples, output_mode=kwargs.get('output_mode','db_phase'),
        freq_min=train_ds.freq_min, freq_max=train_ds.freq_max,
        max_len=kwargs.get('max_len',6), augment=False,
        target_mean=train_ds.target_mean_per_channel, target_std=train_ds.target_std_per_channel,
        hard_freq_indices=hard_freq_indices, freq_range=kwargs.get('freq_range'))
    train_loader = DataLoader(train_ds, batch_size=kwargs.get('batch_size',32), shuffle=True,
        num_workers=kwargs.get('num_workers',0), pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=kwargs.get('batch_size',32), shuffle=False,
        num_workers=kwargs.get('num_workers',0), pin_memory=True, collate_fn=collate_fn)
    stats = {'freq_min': train_ds.freq_min, 'freq_max': train_ds.freq_max,
        'target_mean_per_channel': train_ds.target_mean_per_channel, 'target_std_per_channel': train_ds.target_std_per_channel,
        'device_s_mean': train_ds.device_s_mean, 'device_s_std': train_ds.device_s_std,
        'max_len': kwargs.get('max_len',6)}
    return train_loader, val_loader, train_ds, val_ds, stats


if __name__ == "__main__":
    print("Testing dataset (no type, no n)...")
    data_dir = Path("dataset")
    sub_map = load_all_subdevices(data_dir)
    print(f"Loaded {len(sub_map)} subdevice entries")
    cascade_data = load_cascade_data(data_dir / "cascade", sub_map)
    print(f"Loaded {len(cascade_data)} cascade samples")
    train_loader, val_loader, train_ds, val_ds, stats = build_dataloaders_by_n_split(
        cascade_data, output_mode='db_phase', batch_size=4, freq_range=(0.5, 4.5), max_len=6)
    batch = next(iter(train_loader))
    devices_s, mask, freq_norm, target = batch
    print(f"devices_s: {devices_s.shape}")
    print(f"mask: {mask.shape}, freq: {freq_norm.shape}, target: {target.shape}")
    print(f"stats keys: {list(stats.keys())}")
    print("OK!")
