from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from .data import (
    TrialData,
    body_reference_labels,
    canonical_quat,
    clone_broad,
    downsample_trial,
    get_split_name,
    infer_convention,
    load_all_trials,
    summarize_trials,
)
from .filters import MekfParams, adaptive_sigmas, run_gyro_only, run_mekf
from .geometry import align_quat_sequence_first, normalize_rows, quat_conj, vector_angle
from .metrics import aggregate_by_method, aggregate_groups, orientation_metrics
from .models import (
    PreparedSequence,
    TrainConfig,
    build_features,
    fit_normalizer,
    load_reference_model,
    predict_reference_model,
    train_reference_model,
)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=True)


def prepare_sequences(trials: dict[str, TrialData], convention, mode: str) -> list[PreparedSequence]:
    out: list[PreparedSequence] = []
    for trial in trials.values():
        g_target = normalize_rows(body_reference_labels(trial.quat_gt, convention.gravity_global, convention.body_to_global))
        m_target = None
        if mode == "9d":
            m_target = normalize_rows(body_reference_labels(trial.quat_gt, convention.magnetic_global, convention.body_to_global))
        features = build_features(trial.gyr, trial.acc, trial.mag if mode == "9d" else None, mode)
        raw_g_error = vector_angle(normalize_rows(trial.acc), g_target)
        raw_m_error = None if m_target is None else vector_angle(normalize_rows(trial.mag), m_target)
        out.append(
            PreparedSequence(
                trial.name,
                features,
                g_target.astype(np.float32),
                None if m_target is None else m_target.astype(np.float32),
                raw_g_error.astype(np.float32),
                None if raw_m_error is None else raw_m_error.astype(np.float32),
            )
        )
    return out


def _safe_ahrs_filter(trial: TrialData, mode: str, algorithm: str) -> np.ndarray:
    from ahrs.filters import Madgwick, Mahony

    kwargs = dict(gyr=trial.gyr, acc=trial.acc, frequency=trial.fs)
    if mode == "9d":
        kwargs["mag"] = trial.mag
    if algorithm == "madgwick":
        obj = Madgwick(**kwargs)
    elif algorithm == "mahony":
        obj = Mahony(**kwargs)
    else:
        raise ValueError(algorithm)
    q = np.asarray(obj.Q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise RuntimeError(f"Unexpected {algorithm} quaternion shape {q.shape}")
    bad = ~np.all(np.isfinite(q), axis=1) | (np.linalg.norm(q, axis=1) < 1e-8)
    if np.any(bad):
        first_good = np.flatnonzero(~bad)
        if len(first_good) == 0:
            raise RuntimeError(f"{algorithm} returned no valid quaternion")
        q[bad] = q[first_good[0]]
    return q


def _safe_vqf(trial: TrialData, mode: str) -> np.ndarray:
    from vqf import VQF

    filt = VQF(gyrTs=1.0 / trial.fs)
    if mode == "9d":
        result = filt.updateBatch(
            np.ascontiguousarray(trial.gyr),
            np.ascontiguousarray(trial.acc),
            np.ascontiguousarray(trial.mag),
        )
    else:
        result = filt.updateBatch(
            np.ascontiguousarray(trial.gyr),
            np.ascontiguousarray(trial.acc),
        )
    key = "quat9D" if mode == "9d" else "quat6D"
    return np.asarray(result[key], dtype=np.float64)


def _external_raw(method: str, trial: TrialData, mode: str) -> np.ndarray:
    if method == "VQF":
        return _safe_vqf(trial, mode)
    if method == "Madgwick":
        return _safe_ahrs_filter(trial, mode, "madgwick")
    if method == "Mahony":
        return _safe_ahrs_filter(trial, mode, "mahony")
    raise ValueError(method)


def calibrate_external_inversion(method: str, trial: TrialData, mode: str, q_gt: np.ndarray) -> bool:
    q = _external_raw(method, trial, mode)
    candidates = {False: q, True: quat_conj(q)}
    scores = {}
    n = min(len(q), int(round(20 * trial.fs)))
    movement = np.ones(n, dtype=bool)
    for inv, pred in candidates.items():
        aligned = align_quat_sequence_first(pred[:n], q_gt[:n])
        metric = orientation_metrics(aligned, q_gt[:n], movement, trial.fs)
        key = "inclination_rmse_deg" if mode == "6d" else "total_rmse_deg"
        scores[inv] = metric[key]
    chosen = min(scores, key=scores.get)
    print(f"External convention {method} {mode}: invert={chosen}, scores={scores}")
    return bool(chosen)


def run_external(method: str, trial: TrialData, mode: str, q_gt: np.ndarray, invert: bool) -> np.ndarray:
    q = _external_raw(method, trial, mode)
    if invert:
        q = quat_conj(q)
    return align_quat_sequence_first(q, q_gt)


def ensemble_predict(model_paths: list[Path], features: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    predictions = []
    runtimes = []
    for path in model_paths:
        model, normalizer, payload = load_reference_model(path)
        t0 = time.perf_counter()
        pred = predict_reference_model(model, normalizer, features, payload["window"])
        runtimes.append(time.perf_counter() - t0)
        predictions.append(pred)
    keys = predictions[0].keys()
    out: dict[str, np.ndarray] = {}
    diagnostics: dict[str, float] = {
        "model_count": len(predictions),
        "mean_inference_seconds": float(np.mean(runtimes)),
        "mean_inference_us_per_sample": float(1e6 * np.mean(runtimes) / len(features)),
        "parameter_count": int(sum(p.numel() for p in load_reference_model(model_paths[0])[0].parameters())),
    }
    for key in keys:
        stack = np.stack([p[key] for p in predictions], axis=0)
        if key.endswith("_dir"):
            out[key] = normalize_rows(np.mean(stack, axis=0))
        elif key.endswith("_sigma"):
            branch = key[0]
            dir_key = f"{branch}_dir"
            if dir_key in keys:
                mean_dir = out.get(dir_key)
                if mean_dir is None:
                    dir_stack = np.stack([p[dir_key] for p in predictions], axis=0)
                    mean_dir = normalize_rows(np.mean(dir_stack, axis=0))
                dir_stack = np.stack([p[dir_key] for p in predictions], axis=0)
                spread = vector_angle(
                    dir_stack.reshape(-1, 3),
                    np.broadcast_to(mean_dir, dir_stack.shape).reshape(-1, 3),
                ).reshape(dir_stack.shape[:2])
                out[key] = np.sqrt(np.mean(stack**2 + spread**2, axis=0))
            else:
                out[key] = np.sqrt(np.mean(stack**2, axis=0))
        else:
            out[key] = np.mean(stack, axis=0)
    return out, diagnostics


def reliability_metrics(pred_sigma: np.ndarray, actual_error: np.ndarray, name: str) -> dict[str, float | str]:
    sigma = np.clip(np.asarray(pred_sigma, float), np.deg2rad(0.1), np.deg2rad(90.0))
    err = np.asarray(actual_error, float)
    valid = np.isfinite(sigma) & np.isfinite(err)
    sigma, err = sigma[valid], err[valid]
    if len(err) == 0:
        return {"branch": name, "count": 0}
    corr = stats.spearmanr(sigma, err).statistic
    nll = np.mean(0.5 * ((err / sigma) ** 2 + 2 * np.log(sigma)))
    return {
        "branch": name,
        "count": int(len(err)),
        "spearman_sigma_error": float(corr),
        "nll": float(nll),
        "coverage_1sigma": float(np.mean(err <= sigma)),
        "coverage_2sigma": float(np.mean(err <= 2 * sigma)),
        "mean_sigma_deg": float(np.rad2deg(np.mean(sigma))),
        "mean_actual_error_deg": float(np.rad2deg(np.mean(err))),
    }


def method_predictions(
    trial: TrialData,
    q_gt: np.ndarray,
    convention,
    mode: str,
    ai_models: dict[str, list[Path]],
    seq: PreparedSequence,
    external_invert: dict[tuple[str, str], bool],
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    preds: dict[str, np.ndarray] = {}
    reliability: list[dict] = []
    timing: dict = {}
    preds["Gyro-only"] = run_gyro_only(trial.gyr, q_gt[0], trial.fs, convention.integration_variant)

    base_params = MekfParams()
    raw = run_mekf(
        trial.gyr,
        trial.acc,
        q_gt[0],
        trial.fs,
        convention.gravity_global,
        convention.integration_variant,
        trial.mag if mode == "9d" else None,
        convention.magnetic_global if mode == "9d" else None,
        params=base_params,
    )
    preds["Fixed-MEKF"] = raw["quat"]
    acc_sig, mag_sig = adaptive_sigmas(trial.acc, trial.mag if mode == "9d" else None)
    adaptive = run_mekf(
        trial.gyr,
        trial.acc,
        q_gt[0],
        trial.fs,
        convention.gravity_global,
        convention.integration_variant,
        trial.mag if mode == "9d" else None,
        convention.magnetic_global if mode == "9d" else None,
        acc_sig,
        mag_sig,
        base_params,
    )
    preds["Adaptive-MEKF"] = adaptive["quat"]

    for method in ("Madgwick", "Mahony", "VQF"):
        try:
            preds[method] = run_external(method, trial, mode, q_gt, external_invert[(method, mode)])
        except Exception as exc:
            print(f"WARNING: {method} failed on {trial.name} {mode}: {exc}")

    raw_g = normalize_rows(trial.acc)
    raw_m = normalize_rows(trial.mag) if mode == "9d" else None
    for family, paths in ai_models.items():
        out, diag = ensemble_predict(paths, seq.features)
        timing[f"{family}_{mode}"] = diag
        if family == "TrustNet":
            g_obs = raw_g
            m_obs = raw_m
            g_sigma = out["g_sigma"]
            m_sigma = out.get("m_sigma")
        elif family == "RefNet":
            g_obs = out["g_dir"]
            m_obs = out.get("m_dir")
            g_sigma = np.full(len(g_obs), np.deg2rad(3.0))
            m_sigma = None if m_obs is None else np.full(len(m_obs), np.deg2rad(5.0))
        elif family == "ProbRefNet":
            g_obs = out["g_dir"]
            m_obs = out.get("m_dir")
            g_sigma = out["g_sigma"]
            m_sigma = out.get("m_sigma")
        else:
            raise ValueError(family)
        result = run_mekf(
            trial.gyr,
            g_obs,
            q_gt[0],
            trial.fs,
            convention.gravity_global,
            convention.integration_variant,
            m_obs if mode == "9d" else None,
            convention.magnetic_global if mode == "9d" else None,
            g_sigma,
            m_sigma,
            base_params,
        )
        preds[family + "-MEKF"] = result["quat"]
        if family in ("TrustNet", "ProbRefNet"):
            actual_g = seq.raw_gravity_error if family == "TrustNet" else vector_angle(g_obs, seq.gravity_target)
            reliability.append({"trial": trial.name, "mode": mode, "family": family, **reliability_metrics(g_sigma, actual_g, "gravity")})
            if mode == "9d" and m_sigma is not None and seq.raw_magnetic_error is not None:
                actual_m = seq.raw_magnetic_error if family == "TrustNet" else vector_angle(m_obs, seq.magnetic_target)
                reliability.append({"trial": trial.name, "mode": mode, "family": family, **reliability_metrics(m_sigma, actual_m, "magnetic")})

    oracle = run_mekf(
        trial.gyr,
        seq.gravity_target,
        q_gt[0],
        trial.fs,
        convention.gravity_global,
        convention.integration_variant,
        seq.magnetic_target if mode == "9d" else None,
        convention.magnetic_global if mode == "9d" else None,
        np.full(len(trial.gyr), np.deg2rad(1.0)),
        np.full(len(trial.gyr), np.deg2rad(1.0)) if mode == "9d" else None,
        base_params,
    )
    preds["Oracle-reference-MEKF"] = oracle["quat"]
    return preds, reliability, timing


def train_ai_models(
    sequences_by_mode: dict[str, list[PreparedSequence]],
    output_dir: Path,
    seeds: list[int],
    config: TrainConfig,
) -> tuple[dict[str, dict[str, list[Path]]], list[dict]]:
    model_map: dict[str, dict[str, list[Path]]] = {}
    training_rows: list[dict] = []
    family_mode = {"TrustNet": "trust", "RefNet": "reference", "ProbRefNet": "probabilistic"}
    for mode, sequences in sequences_by_mode.items():
        train_seqs = [s for s in sequences if get_split_name(s.name) == "train"]
        val_seqs = [s for s in sequences if get_split_name(s.name) == "val"]
        normalizer = fit_normalizer(train_seqs)
        model_map[mode] = {}
        for family, task in family_mode.items():
            model_map[mode][family] = []
            for seed in seeds:
                path = output_dir / "models" / f"{family}_{mode}_seed{seed}.pt"
                meta = train_reference_model(task, mode == "9d", train_seqs, val_seqs, normalizer, config, seed, path)
                model_map[mode][family].append(path)
                training_rows.append(
                    {
                        "mode": mode,
                        "family": family,
                        "seed": seed,
                        "best_val": meta["best_val"],
                        "parameter_count": meta["parameter_count"],
                        "epochs_run": len(meta["history"]),
                    }
                )
    return model_map, training_rows


def make_plots(leaderboard: pd.DataFrame, group_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, metric, title in (
        ("6d", "inclination_rmse_deg", "BROAD held-out test: 6-axis inclination RMSE"),
        ("9d", "total_rmse_deg", "BROAD held-out test: 9-axis total orientation RMSE"),
    ):
        df = leaderboard[leaderboard["mode"] == mode].copy()
        df = df[df["method"] != "Oracle-reference-MEKF"].sort_values(metric)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
        ax.barh(df["method"], df[metric])
        ax.invert_yaxis()
        ax.set_xlabel("degrees")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"leaderboard_{mode}.png", dpi=180)
        plt.close(fig)

    for mode, metric in (("6d", "inclination_rmse_deg"), ("9d", "total_rmse_deg")):
        selected_methods = ["VQF", "Adaptive-MEKF", "TrustNet-MEKF", "RefNet-MEKF", "ProbRefNet-MEKF"]
        df = group_df[(group_df["mode"] == mode) & group_df["method"].isin(selected_methods)].copy()
        pivot = df.pivot(index="group", columns="method", values=metric)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(pivot))))
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(f"{mode.upper()} group-wise {'inclination' if mode == '6d' else 'total'} RMSE [deg]")
        fig.colorbar(im, ax=ax, label="degrees")
        fig.tight_layout()
        fig.savefig(out_dir / f"groups_{mode}.png", dpi=180)
        plt.close(fig)


def build_report(summary: dict, convention, leaderboard: pd.DataFrame, reliability: pd.DataFrame, training: pd.DataFrame, out_path: Path) -> None:
    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        use = df[cols].copy()
        for c in cols:
            if pd.api.types.is_float_dtype(use[c]):
                use[c] = use[c].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")
        return use.to_markdown(index=False)

    six = leaderboard[(leaderboard.mode == "6d") & (leaderboard.method != "Oracle-reference-MEKF")].sort_values("inclination_rmse_deg")
    nine = leaderboard[(leaderboard.mode == "9d") & (leaderboard.method != "Oracle-reference-MEKF")].sort_values("total_rmse_deg")
    best6 = six.iloc[0]
    best9 = nine.iloc[0]
    lines = [
        "# Public-data AI-AHRS benchmark",
        "",
        "## Executive findings",
        "",
        f"- The held-out 6-axis winner is **{best6.method}**, with mean trial inclination RMSE **{best6.inclination_rmse_deg:.3f}°**.",
        f"- The held-out 9-axis winner is **{best9.method}**, with mean trial total orientation RMSE **{best9.total_rmse_deg:.3f}°** and heading RMSE **{best9.heading_rmse_deg:.3f}°**.",
        "- Six-axis and nine-axis methods are ranked separately. Six-axis heading is only initial-yaw-aligned relative heading; it is not absolute heading.",
        "- All deployable methods are initialized with the first ground-truth quaternion so that the benchmark isolates propagation and reference-vector robustness rather than initialization.",
        "",
        "## Data and protocol",
        "",
        f"- Dataset: BROAD, {summary['trial_count']} trials, {summary['total_seconds']/60:.1f} minutes total before downsampling.",
        f"- Split: {len(summary['splits']['train'])} train / {len(summary['splits']['val'])} validation / {len(summary['splits']['test'])} held-out test trials.",
        "- Test set holds out fast combined motion, attached-magnet distances, office magnetic environments, and the mixed-disturbance trial.",
        "- All methods use the same downsampled data and trajectory-level split. No windows from a test trajectory enter training.",
        f"- Automatically inferred quaternion convention: body-to-global={convention.body_to_global}; gyro integration={convention.integration_variant}.",
        "",
        "## 6-axis leaderboard",
        "",
        md_table(six, ["method", "inclination_rmse_deg", "inclination_p95_deg", "relative_1s_rmse_deg", "relative_10s_rmse_deg", "heading_drift_deg_min"]),
        "",
        "## 9-axis leaderboard",
        "",
        md_table(nine, ["method", "total_rmse_deg", "heading_rmse_deg", "inclination_rmse_deg", "total_p95_deg", "failure_rate_gt_10deg"]),
        "",
        "## AI variants tested",
        "",
        "- **TrustNet-MEKF:** predicts only accelerometer/magnetometer observation uncertainty and retains raw directions.",
        "- **RefNet-MEKF:** reconstructs clean gravity/magnetic directions but uses fixed uncertainty.",
        "- **ProbRefNet-MEKF:** jointly predicts clean reference directions and time-varying uncertainty.",
        "- **Oracle-reference-MEKF:** uses GT-derived reference directions and is an upper bound, not a deployable competitor.",
        "",
        "## Uncertainty diagnostics",
        "",
    ]
    if reliability.empty:
        lines.append("No uncertainty diagnostics were produced.")
    else:
        rel_agg = reliability.groupby(["mode", "family", "branch"], as_index=False).agg(
            spearman_sigma_error=("spearman_sigma_error", "mean"),
            coverage_1sigma=("coverage_1sigma", "mean"),
            coverage_2sigma=("coverage_2sigma", "mean"),
            mean_sigma_deg=("mean_sigma_deg", "mean"),
            mean_actual_error_deg=("mean_actual_error_deg", "mean"),
        )
        lines.append(md_table(rel_agg, list(rel_agg.columns)))
    lines += [
        "",
        "## Training summary",
        "",
        md_table(training, ["mode", "family", "seed", "best_val", "parameter_count", "epochs_run"]),
        "",
        "## Interpretation boundary",
        "",
        "A 6-axis IMU cannot recover absolute heading without an external heading reference. Any 6-axis yaw result in this benchmark is therefore evaluated only after one initial heading alignment. Likewise, arbitrary external acceleration and arbitrary magnetic disturbance are not identifiable from nine-axis measurements alone; a learned model can exploit temporal priors and should lower confidence when the observation is ambiguous, but it cannot create missing information.",
        "",
        "## Reproducibility",
        "",
        "The artifact contains the exact source code, trained model checkpoints, split manifest, per-trial metrics, aggregate CSV tables, plots, package versions, and workflow logs needed to reproduce the run.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path(".benchmark_work"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-train-windows", type=int, default=80_000)
    parser.add_argument("--max-val-windows", type=int, default=30_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    error_log: list[dict] = []
    broad_root = clone_broad(args.work_dir.resolve() / "broad")
    original_trials, _ = load_all_trials(broad_root)
    original_summary = summarize_trials(original_trials)
    trials = {n: downsample_trial(t, args.target_hz) for n, t in original_trials.items()}
    train_trials = {n: t for n, t in trials.items() if get_split_name(n) == "train"}
    convention = infer_convention(train_trials)
    save_json(
        out_dir / "protocol.json",
        {
            "original_summary": original_summary,
            "downsampled_summary": summarize_trials(trials),
            "target_hz": args.target_hz,
            "window": args.window,
            "seeds": args.seeds,
            "convention": {
                "body_to_global": convention.body_to_global,
                "integration_variant": convention.integration_variant,
                "gravity_global": convention.gravity_global.tolist(),
                "magnetic_global": convention.magnetic_global.tolist(),
            },
        },
    )

    sequences_by_mode = {mode: prepare_sequences(trials, convention, mode) for mode in ("6d", "9d")}
    seq_lookup = {mode: {s.name: s for s in seqs} for mode, seqs in sequences_by_mode.items()}
    config = TrainConfig(window=args.window, epochs=args.epochs, max_train_windows=args.max_train_windows, max_val_windows=args.max_val_windows)
    model_map, training_rows = train_ai_models(sequences_by_mode, out_dir, args.seeds, config)
    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(out_dir / "training_summary.csv", index=False)

    calibration_trial = trials["01_undisturbed_slow_rotation_A"]
    q_cal = canonical_quat(calibration_trial.quat_gt, convention.body_to_global)
    external_invert: dict[tuple[str, str], bool] = {}
    for mode in ("6d", "9d"):
        for method in ("Madgwick", "Mahony", "VQF"):
            try:
                external_invert[(method, mode)] = calibrate_external_inversion(method, calibration_trial, mode, q_cal)
            except Exception as exc:
                external_invert[(method, mode)] = False
                error_log.append({"stage": "calibration", "method": method, "mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})

    trial_rows: list[dict] = []
    reliability_rows: list[dict] = []
    timing_rows: list[dict] = []
    test_names = [n for n in trials if get_split_name(n) == "test"]
    for mode in ("6d", "9d"):
        print(f"Evaluating {mode} on {len(test_names)} held-out trials")
        for name in test_names:
            trial = trials[name]
            q_gt = canonical_quat(trial.quat_gt, convention.body_to_global)
            preds, rel, timing = method_predictions(trial, q_gt, convention, mode, model_map[mode], seq_lookup[mode][name], external_invert)
            reliability_rows.extend(rel)
            for key, value in timing.items():
                timing_rows.append({"trial": name, "mode": mode, "method": key, **value})
            for method, q_pred in preds.items():
                metric = orientation_metrics(q_pred, q_gt, trial.movement, trial.fs)
                trial_rows.append({"trial": name, "mode": mode, "method": method, **metric})

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(out_dir / "test_trial_metrics.csv", index=False)
    leaderboard = pd.DataFrame(aggregate_by_method(trial_rows, "6d") + aggregate_by_method(trial_rows, "9d"))
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    groups = {n: t.groups for n, t in trials.items()}
    group_df = pd.DataFrame(aggregate_groups(trial_rows, groups))
    group_df.to_csv(out_dir / "group_metrics.csv", index=False)
    reliability_df = pd.DataFrame(reliability_rows)
    reliability_df.to_csv(out_dir / "uncertainty_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(out_dir / "runtime_metrics.csv", index=False)
    save_json(out_dir / "errors.json", error_log)

    make_plots(leaderboard, group_df, out_dir / "plots")
    build_report(original_summary, convention, leaderboard, reliability_df, training_df, out_dir / "REPORT.md")
    save_json(
        out_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    )
    print("\nFINAL LEADERBOARD\n", leaderboard.to_string(index=False))
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
