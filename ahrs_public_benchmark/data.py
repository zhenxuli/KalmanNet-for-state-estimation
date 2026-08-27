from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .geometry import (
    ensure_quat_continuity,
    normalize_rows,
    quat_angle,
    quat_conj,
    quat_from_rotvec,
    quat_mul,
    quat_to_rotmat,
    rotate_body_to_global,
    rotate_global_to_body,
)


@dataclass
class TrialData:
    name: str
    gyr: np.ndarray
    acc: np.ndarray
    mag: np.ndarray
    quat_gt: np.ndarray
    movement: np.ndarray
    fs: float
    groups: tuple[str, ...]


@dataclass(frozen=True)
class Convention:
    body_to_global: bool
    integration_variant: str
    gravity_global: np.ndarray
    magnetic_global: np.ndarray


def clone_broad(destination: Path) -> Path:
    destination = Path(destination)
    if (destination / "data_hdf5" / "trials.json").exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/dlaidig/broad.git",
            str(destination),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(destination), "sparse-checkout", "set", "data_hdf5"], check=True)
    return destination


def _orient_rows(a: np.ndarray, width: int) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim == 1 and width == 1:
        return a
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array with width {width}, got {a.shape}")
    if a.shape[1] == width:
        return a
    if a.shape[0] == width:
        return a.T
    raise ValueError(f"Cannot orient array {a.shape} to width {width}")


def _read_h5_value(group: h5py.File, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Missing HDF5 key {key}; available keys: {list(group.keys())}")
    return np.asarray(group[key])


def _read_sampling_rate(f: h5py.File) -> float:
    if "sampling_rate" in f.attrs:
        return float(np.asarray(f.attrs["sampling_rate"]).reshape(-1)[0])
    if "sampling_rate" in f:
        return float(np.asarray(f["sampling_rate"]).reshape(-1)[0])
    # BROAD nominal sampling rate is 2000/7 Hz. Keep this fallback for
    # HDF5 exports that omit the attribute while preserving all signals.
    return 2000.0 / 7.0


def load_trial(path: Path, name: str, groups: Iterable[str]) -> TrialData:
    with h5py.File(path, "r") as f:
        gyr = _orient_rows(_read_h5_value(f, "imu_gyr"), 3).astype(np.float64)
        acc = _orient_rows(_read_h5_value(f, "imu_acc"), 3).astype(np.float64)
        mag = _orient_rows(_read_h5_value(f, "imu_mag"), 3).astype(np.float64)
        quat = _orient_rows(_read_h5_value(f, "opt_quat"), 4).astype(np.float64)
        movement = np.squeeze(_read_h5_value(f, "movement")).astype(bool)
        fs = _read_sampling_rate(f)
    n = min(len(gyr), len(acc), len(mag), len(quat), len(movement))
    gyr, acc, mag, quat, movement = gyr[:n], acc[:n], mag[:n], quat[:n], movement[:n]
    quat = ensure_quat_continuity(quat)
    return TrialData(name, gyr, acc, mag, quat, movement, fs, tuple(groups))


def load_all_trials(root: Path) -> tuple[dict[str, TrialData], dict]:
    data_dir = Path(root) / "data_hdf5"
    with open(data_dir / "trials.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    trials: dict[str, TrialData] = {}
    for name, meta in info["trials"].items():
        trials[name] = load_trial(data_dir / f"{name}.hdf5", name, meta["groups"])
    return trials, info


def _resultant_length(v: np.ndarray) -> float:
    v = normalize_rows(v)
    return float(np.linalg.norm(np.mean(v, axis=0)))


def infer_body_to_global(trials: dict[str, TrialData]) -> bool:
    transformed_a = []
    for trial in trials.values():
        if "undisturbed" not in trial.groups:
            continue
        acc_norm = np.linalg.norm(trial.acc, axis=1)
        med = np.median(acc_norm)
        mask = np.abs(acc_norm - med) < 0.04 * max(med, 1e-6)
        idx = np.flatnonzero(mask)[:: max(1, int(trial.fs // 20))][:3000]
        if len(idx) == 0:
            continue
        R = quat_to_rotmat(trial.quat_gt[idx])
        a = normalize_rows(trial.acc[idx])
        transformed_a.append((R, a))
    if not transformed_a:
        raise RuntimeError("No clean samples available to infer quaternion convention")
    global_h1 = np.concatenate([np.einsum("nij,nj->ni", R, a) for R, a in transformed_a], axis=0)
    global_h2 = np.concatenate([np.einsum("nji,nj->ni", R, a) for R, a in transformed_a], axis=0)
    score_h1 = _resultant_length(global_h1)
    score_h2 = _resultant_length(global_h2)
    print(f"Convention scores: body->global={score_h1:.6f}, global->body={score_h2:.6f}")
    return score_h1 >= score_h2


def _integrate_one(q: np.ndarray, omega_dt: np.ndarray, variant: str) -> np.ndarray:
    sign = -1.0 if variant.endswith("-") else 1.0
    dq = quat_from_rotvec(sign * omega_dt)
    if variant.startswith("right"):
        return quat_mul(q, dq)
    if variant.startswith("left"):
        return quat_mul(dq, q)
    raise ValueError(variant)


def infer_integration_variant(trials: dict[str, TrialData], body_to_global: bool) -> str:
    candidates = ("right+", "right-", "left+", "left-")
    errors: dict[str, list[np.ndarray]] = {c: [] for c in candidates}
    chosen_trials = [t for t in trials.values() if "undisturbed" in t.groups and "rotation" in t.groups][:5]
    for trial in chosen_trials:
        step = max(1, int(trial.fs // 100))
        idx = np.arange(0, len(trial.gyr) - step, step)
        idx = idx[:10000]
        q_seq = trial.quat_gt if body_to_global else quat_conj(trial.quat_gt)
        q0 = q_seq[idx]
        q1 = q_seq[idx + step]
        omega_dt = trial.gyr[idx] * (step / trial.fs)
        for c in candidates:
            q_pred = _integrate_one(q0, omega_dt, c)
            qe = quat_mul(q_pred, quat_conj(q1))
            errors[c].append(quat_angle(qe))
    med = {c: float(np.median(np.concatenate(v))) for c, v in errors.items()}
    print("One-step gyro convention median errors [deg]:", {k: np.rad2deg(v) for k, v in med.items()})
    return min(med, key=med.get)


def infer_global_references(trials: dict[str, TrialData], body_to_global: bool) -> tuple[np.ndarray, np.ndarray]:
    g_samples: list[np.ndarray] = []
    m_samples: list[np.ndarray] = []
    for trial in trials.values():
        if "undisturbed" not in trial.groups:
            continue
        acc_norm = np.linalg.norm(trial.acc, axis=1)
        mag_norm = np.linalg.norm(trial.mag, axis=1)
        acc_med = np.median(acc_norm)
        mag_med = np.median(mag_norm)
        clean = (
            (np.abs(acc_norm - acc_med) < 0.03 * max(acc_med, 1e-6))
            & (np.abs(mag_norm - mag_med) < 0.08 * max(mag_med, 1e-6))
        )
        idx = np.flatnonzero(clean)[:: max(1, int(trial.fs // 20))][:2000]
        if len(idx) == 0:
            continue
        a = normalize_rows(trial.acc[idx])
        m = normalize_rows(trial.mag[idx])
        q = trial.quat_gt[idx]
        if body_to_global:
            g_samples.append(rotate_body_to_global(q, a))
            m_samples.append(rotate_body_to_global(q, m))
        else:
            R = quat_to_rotmat(q)
            g_samples.append(np.einsum("nji,nj->ni", R, a))
            m_samples.append(np.einsum("nji,nj->ni", R, m))
    g = normalize_rows(np.mean(np.concatenate(g_samples, axis=0), axis=0))
    m = normalize_rows(np.mean(np.concatenate(m_samples, axis=0), axis=0))
    print("Inferred global gravity:", g, "magnetic reference:", m)
    return g, m


def infer_convention(trials: dict[str, TrialData]) -> Convention:
    body_to_global = infer_body_to_global(trials)
    integration_variant = infer_integration_variant(trials, body_to_global)
    g, m = infer_global_references(trials, body_to_global)
    return Convention(body_to_global, integration_variant, g, m)


def canonical_quat(q: np.ndarray, body_to_global: bool) -> np.ndarray:
    return q if body_to_global else quat_conj(q)


def body_reference_labels(q: np.ndarray, reference_global: np.ndarray, body_to_global: bool) -> np.ndarray:
    q_bg = canonical_quat(q, body_to_global)
    ref = np.broadcast_to(np.asarray(reference_global, dtype=np.float64), (len(q_bg), 3))
    return rotate_global_to_body(q_bg, ref)


def fixed_split() -> dict[str, set[str]]:
    validation = {
        "19_undisturbed_slow_combined_240s",
        "20_undisturbed_slow_combined_360s",
        "30_disturbed_stationary_magnet_C",
        "31_disturbed_stationary_magnet_D",
    }
    test = {
        "21_undisturbed_fast_combined",
        "22_undisturbed_fast_combined_240s",
        "23_undisturbed_fast_combined_360s",
        "35_disturbed_attached_magnet_4cm",
        "36_disturbed_attached_magnet_5cm",
        "37_disturbed_office_A",
        "38_disturbed_office_B",
        "39_disturbed_mixed",
    }
    return {"val": validation, "test": test}


def get_split_name(trial_name: str) -> str:
    split = fixed_split()
    if trial_name in split["val"]:
        return "val"
    if trial_name in split["test"]:
        return "test"
    return "train"


def downsample_trial(trial: TrialData, target_hz: float = 50.0) -> TrialData:
    stride = max(1, int(round(trial.fs / target_hz)))
    sl = slice(None, None, stride)
    return TrialData(
        name=trial.name,
        gyr=trial.gyr[sl],
        acc=trial.acc[sl],
        mag=trial.mag[sl],
        quat_gt=trial.quat_gt[sl],
        movement=trial.movement[sl],
        fs=trial.fs / stride,
        groups=trial.groups,
    )


def summarize_trials(trials: dict[str, TrialData]) -> dict:
    total_samples = sum(len(t.gyr) for t in trials.values())
    total_seconds = sum(len(t.gyr) / t.fs for t in trials.values())
    return {
        "trial_count": len(trials),
        "total_samples": int(total_samples),
        "total_seconds": float(total_seconds),
        "sampling_rates": sorted({round(t.fs, 6) for t in trials.values()}),
        "splits": {
            s: [n for n in trials if get_split_name(n) == s] for s in ("train", "val", "test")
        },
    }
