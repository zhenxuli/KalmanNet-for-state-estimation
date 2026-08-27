from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .geometry import normalize_rows


@dataclass
class PreparedSequence:
    name: str
    features: np.ndarray
    gravity_target: np.ndarray
    magnetic_target: np.ndarray | None
    raw_gravity_error: np.ndarray
    raw_magnetic_error: np.ndarray | None


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, x: dict) -> "FeatureNormalizer":
        return cls(np.asarray(x["mean"], float), np.asarray(x["std"], float))


def build_features(gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None, mode: str) -> np.ndarray:
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    gyr_norm = np.linalg.norm(gyr, axis=1, keepdims=True)
    pieces = [gyr, acc, acc_norm, gyr_norm]
    if mode == "9d":
        if mag is None:
            raise ValueError("9D features require magnetometer")
        mag_norm = np.linalg.norm(mag, axis=1, keepdims=True)
        pieces = [gyr, acc, mag, acc_norm, mag_norm, gyr_norm]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def fit_normalizer(sequences: list[PreparedSequence], max_samples: int = 300_000) -> FeatureNormalizer:
    x = np.concatenate([s.features for s in sequences], axis=0)
    if len(x) > max_samples:
        rng = np.random.default_rng(1234)
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return FeatureNormalizer(mean.astype(np.float32), std.astype(np.float32))


class ReferenceWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[PreparedSequence],
        normalizer: FeatureNormalizer,
        window: int,
        max_windows: int | None,
        seed: int,
    ) -> None:
        self.sequences = sequences
        self.normalizer = normalizer
        self.window = int(window)
        indices: list[tuple[int, int]] = []
        for si, seq in enumerate(sequences):
            indices.extend((si, k) for k in range(len(seq.features)))
        if max_windows is not None and len(indices) > max_windows:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(indices), size=max_windows, replace=False)
            indices = [indices[int(i)] for i in chosen]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        si, k = self.indices[idx]
        seq = self.sequences[si]
        start = k - self.window + 1
        if start >= 0:
            x = seq.features[start : k + 1]
        else:
            pad = np.repeat(seq.features[[0]], -start, axis=0)
            x = np.concatenate([pad, seq.features[: k + 1]], axis=0)
        x = self.normalizer.transform(x).T.astype(np.float32)
        item = {
            "x": torch.from_numpy(x),
            "g": torch.from_numpy(seq.gravity_target[k].astype(np.float32)),
            "g_err": torch.tensor(seq.raw_gravity_error[k], dtype=torch.float32),
        }
        if seq.magnetic_target is not None:
            item["m"] = torch.from_numpy(seq.magnetic_target[k].astype(np.float32))
            item["m_err"] = torch.tensor(seq.raw_magnetic_error[k], dtype=torch.float32)
        return item


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad:
            y = y[..., :-self.pad]
        return y


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, 3, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, 3, dilation),
            nn.GELU(),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList([ResidualTCNBlock(hidden, d) for d in (1, 2, 4, 8)])
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h[..., -1])


class ReferenceModel(nn.Module):
    def __init__(self, in_channels: int, mode: Literal["trust", "reference", "probabilistic"], has_mag: bool) -> None:
        super().__init__()
        self.mode = mode
        self.has_mag = has_mag
        self.encoder = TemporalEncoder(in_channels)
        branches = 2 if has_mag else 1
        if mode == "trust":
            out_dim = branches
        elif mode == "reference":
            out_dim = 3 * branches
        elif mode == "probabilistic":
            out_dim = 4 * branches
        else:
            raise ValueError(mode)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, out_dim))

    @staticmethod
    def _unit(v: torch.Tensor) -> torch.Tensor:
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-6)

    @staticmethod
    def _sigma(logit: torch.Tensor) -> torch.Tensor:
        min_sigma = math.radians(0.5)
        max_sigma = math.radians(80.0)
        return torch.clamp(torch.nn.functional.softplus(logit) + min_sigma, max=max_sigma)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.head(self.encoder(x))
        out: dict[str, torch.Tensor] = {}
        if self.mode == "trust":
            out["g_sigma"] = self._sigma(y[:, 0])
            if self.has_mag:
                out["m_sigma"] = self._sigma(y[:, 1])
        elif self.mode == "reference":
            out["g_dir"] = self._unit(y[:, :3])
            if self.has_mag:
                out["m_dir"] = self._unit(y[:, 3:6])
        else:
            out["g_dir"] = self._unit(y[:, :3])
            out["g_sigma"] = self._sigma(y[:, 3])
            if self.has_mag:
                out["m_dir"] = self._unit(y[:, 4:7])
                out["m_sigma"] = self._sigma(y[:, 7])
        return out


def _angular_error_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.clamp(torch.linalg.norm(a, dim=-1, keepdim=True), min=1e-6)
    b = b / torch.clamp(torch.linalg.norm(b, dim=-1, keepdim=True), min=1e-6)
    dot = torch.sum(a * b, dim=-1)
    return torch.acos(torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6))


def model_loss(model: ReferenceModel, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    x = batch["x"].to(device)
    out = model(x)
    losses: list[torch.Tensor] = []
    if model.mode == "trust":
        g_err = batch["g_err"].to(device)
        sg = out["g_sigma"]
        losses.append(0.5 * ((g_err / sg) ** 2 + 2.0 * torch.log(sg)))
        if model.has_mag:
            m_err = batch["m_err"].to(device)
            sm = out["m_sigma"]
            losses.append(0.5 * ((m_err / sm) ** 2 + 2.0 * torch.log(sm)))
    elif model.mode == "reference":
        g = batch["g"].to(device)
        losses.append(1.0 - torch.sum(out["g_dir"] * g, dim=-1))
        if model.has_mag:
            m = batch["m"].to(device)
            losses.append(1.0 - torch.sum(out["m_dir"] * m, dim=-1))
    else:
        g = batch["g"].to(device)
        eg = _angular_error_torch(out["g_dir"], g)
        sg = out["g_sigma"]
        losses.append(0.5 * ((eg / sg) ** 2 + 2.0 * torch.log(sg)) + 0.1 * (1.0 - torch.sum(out["g_dir"] * g, dim=-1)))
        if model.has_mag:
            m = batch["m"].to(device)
            em = _angular_error_torch(out["m_dir"], m)
            sm = out["m_sigma"]
            losses.append(0.5 * ((em / sm) ** 2 + 2.0 * torch.log(sm)) + 0.1 * (1.0 - torch.sum(out["m_dir"] * m, dim=-1)))
    return torch.stack(losses, dim=0).mean()


@dataclass
class TrainConfig:
    window: int = 64
    hidden: int = 32
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_train_windows: int = 80_000
    max_val_windows: int = 30_000
    patience: int = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def train_reference_model(
    mode: Literal["trust", "reference", "probabilistic"],
    has_mag: bool,
    train_sequences: list[PreparedSequence],
    val_sequences: list[PreparedSequence],
    normalizer: FeatureNormalizer,
    config: TrainConfig,
    seed: int,
    out_path: Path,
) -> dict:
    seed_everything(seed)
    device = torch.device("cpu")
    in_channels = train_sequences[0].features.shape[1]
    model = ReferenceModel(in_channels, mode, has_mag).to(device)
    train_ds = ReferenceWindowDataset(train_sequences, normalizer, config.window, config.max_train_windows, seed)
    val_ds = ReferenceWindowDataset(val_sequences, normalizer, config.window, config.max_val_windows, seed + 1000)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict] = []
    bad_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {mode}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(float(model_loss(model, batch, device)))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"{mode} {'9D' if has_mag else '6D'} seed={seed} epoch={epoch+1}: {row}")
        if row["val_loss"] < best_val - 1e-5:
            best_val = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mode": mode,
            "has_mag": has_mag,
            "in_channels": in_channels,
            "window": config.window,
            "normalizer": normalizer.to_json(),
            "seed": seed,
            "best_val": best_val,
            "history": history,
        },
        out_path,
    )
    return {"best_val": best_val, "history": history, "parameter_count": sum(p.numel() for p in model.parameters())}


def load_reference_model(path: Path) -> tuple[ReferenceModel, FeatureNormalizer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ReferenceModel(payload["in_channels"], payload["mode"], payload["has_mag"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, FeatureNormalizer.from_json(payload["normalizer"]), payload


def predict_reference_model(
    model: ReferenceModel,
    normalizer: FeatureNormalizer,
    features: np.ndarray,
    window: int,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    n = len(features)
    outputs: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for start_k in range(0, n, batch_size):
            ks = np.arange(start_k, min(n, start_k + batch_size))
            x_batch = []
            for k in ks:
                start = k - window + 1
                if start >= 0:
                    x = features[start : k + 1]
                else:
                    pad = np.repeat(features[[0]], -start, axis=0)
                    x = np.concatenate([pad, features[: k + 1]], axis=0)
                x_batch.append(normalizer.transform(x).T.astype(np.float32))
            x_t = torch.from_numpy(np.stack(x_batch, axis=0))
            out = model(x_t)
            for key, value in out.items():
                outputs.setdefault(key, []).append(value.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outputs.items()}
