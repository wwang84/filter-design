#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loss.py - Custom loss functions for cascade correction model

Components:
- Huber loss with amplitude-dependent weighting
- Port weighting (reflection ports weighted higher)
- Frequency weighting based on target variance
- Reciprocity penalty (S21 ~ S12)
- Selective smoothness penalty (reduced at resonant edges)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReciprocalLoss(nn.Module):
    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight

    def forward(self, pred_denorm, n_freq):
        B = pred_denorm.shape[0]
        p = pred_denorm.view(B, 4, n_freq)
        return self.weight * F.l1_loss(p[:, 1], p[:, 3], reduction='none').mean(dim=1)


class SelectiveSmoothnessLoss(nn.Module):
    """Selective smoothness penalty: reduced at frequencies with rapid magnitude changes"""
    def __init__(self, weight=0.05, s11_smooth_scale=0.15):
        super().__init__()
        self.weight = weight
        self.s11_smooth_scale = s11_smooth_scale

    def forward(self, pred_denorm, n_freq, casc_flat):
        B = pred_denorm.shape[0]
        p = pred_denorm.view(B, 4, n_freq)
        c = casc_flat.view(B, 8, n_freq)

        s11_mag = torch.sqrt(c[:, 0]**2 + c[:, 1]**2 + 1e-12)
        s21_mag = torch.sqrt(c[:, 2]**2 + c[:, 3]**2 + 1e-12)
        s22_mag = torch.sqrt(c[:, 4]**2 + c[:, 5]**2 + 1e-12)
        s12_mag = torch.sqrt(c[:, 6]**2 + c[:, 7]**2 + 1e-12)

        def _grad(mag):
            g = torch.abs(mag[:, 1:] - mag[:, :-1])
            g = F.pad(g, (1, 0), value=0)
            return g / (g.max(dim=-1, keepdim=True)[0] + 1e-8)

        grad = torch.stack([_grad(s11_mag), _grad(s21_mag), _grad(s22_mag), _grad(s12_mag)], dim=1)
        smooth_mask = 1.0 / (1.0 + 5.0 * grad)

        diff1 = p[:, :, 1:] - p[:, :, :-1]
        diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
        smooth_mask = smooth_mask[:, :, 1:-1]

        port_smooth_scale = torch.ones(4, device=p.device)
        port_smooth_scale[0] = self.s11_smooth_scale   # S11
        port_smooth_scale[2] = self.s11_smooth_scale   # S22
        smooth_mask = smooth_mask * port_smooth_scale.view(1, 4, 1)

        return self.weight * (diff2**2 * smooth_mask).mean(dim=[1, 2])


class TotalLoss(nn.Module):
    """Composite loss: Huber + port weighting + frequency weighting + reciprocity + smoothness"""

    def __init__(
        self,
        huber_beta: float = 1.0,
        amplitude_weight_strength: float = 2.0,
        reciprocal_weight: float = 0.2,
        smooth_weight: float = 0.05,
        channel_std=None,
        channel_mean=None,
        freq_std=None,
        inv_var_power: float = 0.5,
        device_weight_power: float = 1.0,
        port_weights=None,
        s11_smooth_scale: float = 0.15,
    ):
        super().__init__()
        self.huber_beta = huber_beta
        self.amp_strength = amplitude_weight_strength
        self.recip_weight = reciprocal_weight
        self.smooth_weight = smooth_weight
        self.device_weight_power = device_weight_power
        self.s11_smooth_scale = s11_smooth_scale

        if port_weights is not None:
            self.register_buffer('port_weights', torch.tensor(port_weights, dtype=torch.float32))
        else:
            self.register_buffer('port_weights', torch.tensor([2.5, 1.0, 2.5, 1.0], dtype=torch.float32))

        if channel_mean is not None:
            self.register_buffer('channel_mean', channel_mean)
        else:
            self.channel_mean = None
        if channel_std is not None:
            self.register_buffer('channel_std', channel_std)
        else:
            self.channel_std = None

        if freq_std is not None:
            fw = 1.0 / (1.0 + freq_std) ** inv_var_power
            fw = fw / fw.mean()
            self.register_buffer('freq_weight', fw)
        else:
            self.freq_weight = None

        self.reciprocal_loss = ReciprocalLoss(reciprocal_weight)
        self.smooth_loss = SelectiveSmoothnessLoss(smooth_weight, s11_smooth_scale)
        self.register_buffer('mean_num_devices', torch.tensor(1.0))

    def forward(self, pred, target, n_freq, num_devices, x=None):
        B = pred.shape[0]
        p = pred.view(B, 4, n_freq)
        t = target.view(B, 4, n_freq)

        if self.channel_mean is not None and self.channel_std is not None:
            mean = self.channel_mean[:4].to(pred.device).view(1, 4, 1)
            std = self.channel_std[:4].to(pred.device).view(1, 4, 1)
            pd = p * std + mean
            td = t * std + mean
        else:
            pd, td = p, t

        # Huber loss
        err = pd - td
        abs_err = torch.abs(err)
        huber = torch.where(abs_err < self.huber_beta,
                            0.5 * err**2 / self.huber_beta,
                            abs_err - 0.5 * self.huber_beta)

        # Amplitude weighting (higher weight at low amplitudes)
        if x is not None:
            casc = x[:, :8*n_freq].view(B, 8, n_freq)
            s11_mag = torch.sqrt(casc[:, 0]**2 + casc[:, 1]**2 + 1e-12)
            s21_mag = torch.sqrt(casc[:, 2]**2 + casc[:, 3]**2 + 1e-12)
            s22_mag = torch.sqrt(casc[:, 4]**2 + casc[:, 5]**2 + 1e-12)
            s12_mag = torch.sqrt(casc[:, 6]**2 + casc[:, 7]**2 + 1e-12)
            mag = torch.stack([s11_mag, s21_mag, s22_mag, s12_mag], dim=1)
            amp_weight = 1.0 + self.amp_strength * (1.0 - mag)
            amp_weight = torch.clamp(amp_weight, 1.0, 3.0)
        else:
            amp_weight = torch.ones_like(huber)

        loss = huber * amp_weight

        # Port weighting
        pw = self.port_weights.to(pred.device).view(1, 4, 1)
        loss = loss * pw

        # Frequency weighting
        if self.freq_weight is not None:
            loss = loss * self.freq_weight.to(pred.device).view(1, 1, -1)

        direct = loss.mean(dim=[1, 2])

        # Reciprocity
        rec = self.reciprocal_loss(pd.view(B, 4*n_freq), n_freq)

        # Selective smoothness
        if x is not None:
            sm = self.smooth_loss(pd.view(B, 4*n_freq), n_freq, x)
        else:
            sm = torch.zeros_like(rec)

        total = direct + self.recip_weight * rec + self.smooth_weight * sm

        if self.device_weight_power != 0 and num_devices is not None:
            mean_d = self.mean_num_devices.to(pred.device)
            w = (num_devices / mean_d) ** self.device_weight_power
            total = total * (w / w.mean())

        db_mae = abs_err.mean().detach()
        return total.mean(), db_mae, rec.mean(), sm.mean(), direct.mean()
