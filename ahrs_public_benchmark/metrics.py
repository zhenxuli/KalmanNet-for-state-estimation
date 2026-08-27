from __future__ import annotations

import numpy as np

from .geometry import (
    calculate_heading_inclination_error,
    calculate_total_error,
    quat_conj,
    quat_mul,
    signed_heading_error,
)


def _summary(x_rad: np.ndarray, prefix: str) -> dict[str, float]:
    x_deg = np.rad2deg(x_rad[np.isfinite(x_rad)])
    if len(x_deg) == 0:
        return {f"{prefix}_rmse_deg": np.nan, f"{prefix}_median_deg": np.nan, f"{prefix}_p95_deg": np.nan}
    return {
        f"{prefix}_rmse_deg": float(np.sqrt(np.mean(x_deg**2))),
        f"{prefix}_mean_deg": float(np.mean(x_deg)),
        f"{prefix}_median_deg": float(np.median(x_deg)),
        f"{prefix}_p95_deg": float(np.percentile(x_deg, 95)),
        f"{prefix}_max_deg": float(np.max(x_deg)),
    }


def relative_rotation_error(q_est: np.ndarray, q_gt: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or len(q_est) <= lag:
        return np.empty(0)
    rel_est = quat_mul(quat_conj(q_est[:-lag]), q_est[lag:])
    rel_gt = quat_mul(quat_conj(q_gt[:-lag]), q_gt[lag:])
    return calculate_total_error(rel_est, rel_gt)


def orientation_metrics(q_est: np.ndarray, q_gt: np.ndarray, movement: np.ndarray, fs: float) -> dict[str, float]:
    n = min(len(q_est), len(q_gt), len(movement))
    q_est, q_gt, movement = q_est[:n], q_gt[:n], movement[:n].astype(bool)
    valid = movement & np.all(np.isfinite(q_est), axis=1) & np.all(np.isfinite(q_gt), axis=1)
    valid[: min(n, int(round(fs)))] = False
    total = calculate_total_error(q_est, q_gt)
    heading, inclination = calculate_heading_inclination_error(q_est, q_gt)
    out: dict[str, float] = {}
    out.update(_summary(total[valid], "total"))
    out.update(_summary(heading[valid], "heading"))
    out.update(_summary(inclination[valid], "inclination"))
    total_deg = np.rad2deg(total[valid])
    for threshold in (5, 10, 20):
        out[f"failure_rate_gt_{threshold}deg"] = float(np.mean(total_deg > threshold)) if len(total_deg) else np.nan

    signed = np.unwrap(signed_heading_error(q_est, q_gt))
    idx = np.flatnonzero(valid)
    if len(idx) >= 10:
        t_min = idx / fs / 60.0
        slope = np.polyfit(t_min, np.rad2deg(signed[idx]), 1)[0]
        out["heading_drift_deg_min"] = float(slope)
    else:
        out["heading_drift_deg_min"] = np.nan

    for seconds in (1.0, 10.0):
        lag = max(1, int(round(seconds * fs)))
        rel = relative_rotation_error(q_est, q_gt, lag)
        if len(rel):
            pair_mask = valid[:-lag] & valid[lag:]
            out.update(_summary(rel[pair_mask], f"relative_{int(seconds)}s"))
        else:
            out[f"relative_{int(seconds)}s_rmse_deg"] = np.nan
    return out


def aggregate_by_method(rows: list[dict], mode: str) -> list[dict]:
    methods = sorted({r["method"] for r in rows if r["mode"] == mode})
    out = []
    for method in methods:
        selected = [r for r in rows if r["mode"] == mode and r["method"] == method]
        metric_keys = sorted(k for k in selected[0] if k.endswith("_deg") or k.endswith("_deg_min") or k.startswith("failure_rate"))
        row = {"mode": mode, "method": method, "trial_count": len(selected)}
        for key in metric_keys:
            vals = np.asarray([r.get(key, np.nan) for r in selected], float)
            row[key] = float(np.nanmean(vals))
        out.append(row)
    return out


def aggregate_groups(rows: list[dict], trial_groups: dict[str, tuple[str, ...]]) -> list[dict]:
    all_groups = sorted({g for gs in trial_groups.values() for g in gs})
    result = []
    for mode in sorted({r["mode"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["mode"] == mode}):
            method_rows = [r for r in rows if r["mode"] == mode and r["method"] == method]
            for group in all_groups:
                selected = [r for r in method_rows if group in trial_groups[r["trial"]]]
                if not selected:
                    continue
                metric = "inclination_rmse_deg" if mode == "6d" else "total_rmse_deg"
                result.append(
                    {
                        "mode": mode,
                        "method": method,
                        "group": group,
                        "trial_count": len(selected),
                        metric: float(np.mean([r[metric] for r in selected])),
                    }
                )
    return result
