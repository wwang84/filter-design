#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py - Cascade correction model

Architecture:
1. PerDeviceEncoder: residual Conv1d + GroupNorm (9ch input: 8ch RI + 1ch port_width_diff)
2. FreqSelfAttention: frequency-axis Transformer (Pre-LN, multi-layer)
3. DeviceCrossAttention: device-axis cross-attention (bidirectional device <-> freq)
4. FreqSineEncoder: learnable sine encoding for frequency features
5. Fusion + Head: multi-level fusion Conv1d -> 4-channel output (S11,S21,S22,S12 dB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    def forward(self, x): return x + self.pos_embedding[:, :x.size(1), :]


class ResidualConvBlock(nn.Module):
    """Conv1d + GroupNorm + GELU + residual connection"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1, stride, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x))) + self.skip(x)


class PerDeviceEncoder(nn.Module):
    """S-parameter encoder (9 channels: 8ch RI + 1ch port_width_diff), 3-layer residual Conv1d"""
    def __init__(self, n_freq, d_model=320, in_channels=9):
        super().__init__()
        self.n_freq = n_freq
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 96, 7, padding=3, bias=False),
            nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.block1 = ResidualConvBlock(96, 192, 5)
        self.block2 = ResidualConvBlock(192, d_model, 3)

    def forward(self, s):
        x = self.stem(s)
        x = self.block1(x)
        x = self.block2(x)
        return x


class FreqSelfAttention(nn.Module):
    """Frequency-axis self-attention (Pre-LN)"""
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=200)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        x = self.out_proj(x)
        return x.permute(0, 2, 1)


class DeviceCrossAttention(nn.Module):
    """Device-axis cross-attention (Pre-LN)"""
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_ff=512, dropout=0.1, max_len=6):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
        self.layers = nn.ModuleList(layers)
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, d_model))

    def forward(self, x, mask):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        mean = (x * m).sum(1) / (m.sum(1) + 1e-8)
        masked = x.masked_fill(~m.bool(), float('-inf'))
        max_val, _ = masked.max(1)
        return self.pool(mean + max_val) + x.mean(dim=1)


class FreqSineEncoder(nn.Module):
    """Sine encoding with learnable frequencies, captures high-frequency features"""
    def __init__(self, n_freq, out_dim=128, n_harmonics=8):
        super().__init__()
        self.freqs = nn.Parameter(torch.linspace(0.5, 15.0, n_harmonics).view(1, 1, -1))
        self.phases = nn.Parameter(torch.zeros(1, 1, n_harmonics))
        in_dim = n_freq * (1 + 2 * n_harmonics)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim))

    def forward(self, x):
        B, F = x.shape
        f_scaled = x.unsqueeze(-1) * self.freqs
        sin_feat = torch.sin(2 * math.pi * f_scaled + self.phases)
        cos_feat = torch.cos(2 * math.pi * f_scaled + self.phases)
        all_feat = torch.cat([x, sin_feat.flatten(1), cos_feat.flatten(1)], dim=-1)
        return self.proj(all_feat)


class ResidualNet(nn.Module):
    """
    Cascade correction model
    - Residual conv encoder (9ch input)
    - Frequency-axis self-attention
    - Device-axis cross-attention
    - Sine frequency encoding
    - Multi-level fusion + Conv1d head
    """

    def __init__(self, n_freq: int, max_len: int = 6,
                 d_model: int = 320, nhead: int = 8,
                 num_device_layers: int = 4, num_freq_layers: int = 4,
                 dim_ff: int = 768, dropout: float = 0.15, attn_dropout: float = 0.05):
        super().__init__()
        self.n_freq = n_freq
        self.max_len = max_len
        self.d_model = d_model

        self.device_encoder = PerDeviceEncoder(n_freq, d_model, in_channels=9)
        self.freq_self_attn = FreqSelfAttention(d_model, nhead, num_freq_layers, dim_ff, attn_dropout)
        self.device_cross_attn = DeviceCrossAttention(d_model, nhead, num_device_layers, dim_ff, attn_dropout, max_len)
        self.freq_encoder = FreqSineEncoder(n_freq, 128)

        self.fusion = nn.Sequential(
            nn.Conv1d(d_model + 128, 384, 3, padding=1, bias=False),
            nn.GroupNorm(8, 384), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(384, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 4, 3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, devices_s, mask, freq_norm):
        """
        Args:
            devices_s: (B, L, 9, F) normalized subdevice S-params
            mask:      (B, L) padding mask
            freq_norm: (B, F) normalized frequencies
        Returns:
            s_pred: (B, 4*F) predicted dB S-params
        """
        B, L, n_ch = devices_s.shape[:3]

        d_s = devices_s.view(B * L, n_ch, self.n_freq)
        feats = self.device_encoder(d_s)

        feats = self.freq_self_attn(feats)
        feats = feats.view(B, L, self.d_model, self.n_freq)

        feats_t = feats.permute(0, 3, 1, 2).reshape(B * self.n_freq, L, self.d_model)
        mask_t = mask.unsqueeze(1).expand(B, self.n_freq, L).reshape(B * self.n_freq, L)
        global_feat = self.device_cross_attn(feats_t, mask_t)
        global_feat = global_feat.view(B, self.n_freq, self.d_model).permute(0, 2, 1)

        freq_feat = self.freq_encoder(freq_norm).unsqueeze(-1).expand(-1, -1, self.n_freq)

        fused = torch.cat([global_feat, freq_feat], dim=1)
        fused = self.fusion(fused)
        s_pred = self.head(fused).view(B, 4 * self.n_freq)
        return s_pred


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    B, L, F = 4, 6, 146
    # m = ResidualNet(n_freq=F).to(dev)
    # print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

    ds = torch.randn(B, L, 9, F).to(dev)
    mk = torch.zeros(B, L, dtype=torch.bool).to(dev)
    mk[0, :2] = mk[1, :3] = mk[2, :4] = mk[3, :6] = True
    fn = torch.rand(B, F).to(dev)

    y = m(ds, mk, fn)
    print(f"Output: {y.shape}")
    y.mean().backward()
    print("Model OK!")
