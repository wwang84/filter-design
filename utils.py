#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - V4 utility functions: checkpoint, logging, plotting, EMA, etc.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import deque
import shutil

# ==================== Checkpoint Management ====================
def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss: float,
    filepath: Path,
    ema_model: Optional[torch.nn.Module] = None,
    best_loss: Optional[float] = None
):
    """Save training checkpoint"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'best_loss': best_loss if best_loss is not None else loss,
    }
    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    device: str = 'cpu',
    ema_model: Optional[torch.nn.Module] = None
) -> Dict:
    """Load checkpoint"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    return checkpoint


def save_model_with_stats(
    model: torch.nn.Module,
    stats: Dict,
    model_path: Path,
    stats_path: Path
):
    """Save model weights and statistics"""
    model_path = Path(model_path)
    stats_path = Path(stats_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


def load_model_with_stats(
    model: torch.nn.Module,
    model_path: Path,
    stats_path: Path,
    device: str = 'cpu'
) -> Dict:
    """Load model weights and statistics"""
    model_path = Path(model_path)
    stats_path = Path(stats_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return stats


# ==================== EMA (Exponential Moving Average) ====================
class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ==================== Training Logger ====================
class TrainLogger:
    """Training log recorder"""
    def __init__(self, log_dir: Path, log_name: str = "train_log.json"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_name
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'best_val_loss': []
        }
        # Load existing log
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                self.history = json.load(f)

    def log(self, epoch: int, train_loss: float, val_loss: float, lr: float, best_val_loss: float):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(lr)
        self.history['best_val_loss'].append(best_val_loss)
        self.save()

    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self):
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                self.history = json.load(f)
        return self.history

    def plot_loss(self, save_path: Optional[Path] = None):
        """Plot loss curves"""
        if not self.history['epoch']:
            print("No data to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss', alpha=0.7)
        axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training/Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate curve
        axes[1].plot(self.history['epoch'], self.history['lr'], label='Learning Rate', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ==================== Learning Rate Scheduler ====================
class CosineWarmRestart:
    """
    Cosine annealing with warm restarts
    Equivalent to torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.T_i = T_0
        self.step()

    def step(self):
        self.last_epoch += 1
        if self.last_epoch == self.T_i:
            self.T_i = int(self.T_i * self.T_mult)
            self.last_epoch = 0
        progress = self.last_epoch / self.T_i
        lr = self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def state_dict(self):
        return {
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
            'T_i': self.T_i,
        }

    def load_state_dict(self, state_dict):
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self.T_i = state_dict['T_i']
        # Restore current learning rate (need to recalculate)
        progress = self.last_epoch / self.T_i
        lr = self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def create_lr_scheduler(optimizer, config: Dict):
    """Create learning rate scheduler based on config"""
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        return CosineWarmRestart(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5),
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.5)
        )
    else:
        return None


# ==================== Other Utilities ====================
def set_seed(seed: int = 42):
    """Set random seed"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: Path):
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Get available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def format_time(seconds: float) -> str:
    """Format time string"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    hours = minutes // 60
    minutes = minutes % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"


def dict_to_device(data: Dict, device: str):
    """Move all tensors in dict to specified device"""
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    return data


def flatten_sparams_to_tensor(S: np.ndarray) -> np.ndarray:
    """Flatten S-parameter matrix to vector (8*n_freq)"""
    n_freq = S.shape[0]
    flat = np.zeros(8 * n_freq, dtype=np.float32)
    flat[0*n_freq:1*n_freq] = S[:,0,0].real
    flat[1*n_freq:2*n_freq] = S[:,0,0].imag
    flat[2*n_freq:3*n_freq] = S[:,1,0].real
    flat[3*n_freq:4*n_freq] = S[:,1,0].imag
    flat[4*n_freq:5*n_freq] = S[:,1,1].real
    flat[5*n_freq:6*n_freq] = S[:,1,1].imag
    flat[6*n_freq:7*n_freq] = S[:,0,1].real
    flat[7*n_freq:8*n_freq] = S[:,0,1].imag
    return flat


if __name__ == "__main__":
    print("Utils module loaded successfully.")
