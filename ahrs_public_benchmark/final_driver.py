from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    TrialData, body_reference_labels, canonical_quat, clone_broad, downsample_trial,
    get_split_name, infer_convention, load_all_trials, summarize_trials,
)
from .filters import MekfParams, adaptive_sigmas, run_gyro_only, run_mekf
from .geometry import align_quat_sequence_first, normalize_rows, vector_angle
from .metrics import aggregate_by_method, aggregate_groups, orientation_metrics
from .models import PreparedSequence, TrainConfig
from .run_benchmark import (
    ensemble_predict, make_plots, prepare_sequences, reliability_metrics,
    save_json, train_ai_models,
)


def first_valid_crop(trial: TrialData, q_gt: np.ndarray, seq: PreparedSequence):
    idx = np.flatnonzero(np.all(np.isfinite(q_gt), axis=1))
    if len(idx) == 0:
        raise ValueError(f"no finite optical GT for {trial.name}")
    k0 = int(idx[0])
    t = TrialData(
        trial.name, trial.gyr[k0:], trial.acc[k0:], trial.mag[k0:],
        trial.quat_gt[k0:], trial.movement[k0:], trial.fs, trial.groups,
    )
    s = PreparedSequence(
        seq.name, seq.features[k0:], seq.gravity_target[k0:],
        None if seq.magnetic_target is None else seq.magnetic_target[k0:],
        seq.raw_gravity_error[k0:],
        None if seq.raw_magnetic_error is None else seq.raw_magnetic_error[k0:],
    )
    return t, q_gt[k0:], s, k0


def run_madgwick(trial: TrialData, mode: str, q0: np.ndarray) -> np.ndarray:
    from ahrs.filters import Madgwick
    gain = 0.12 if mode == "9d" else 0.033
    f = Madgwick(frequency=trial.fs, gain=gain)
    q = np.zeros((len(trial.gyr), 4), float)
    q[0] = q0 / np.linalg.norm(q0)
    for k in range(1, len(q)):
        if mode == "9d":
            q[k] = f.updateMARG(q[k-1], trial.gyr[k], trial.acc[k], trial.mag[k])
        else:
            q[k] = f.updateIMU(q[k-1], trial.gyr[k], trial.acc[k])
    return q


def run_mahony(trial: TrialData, mode: str, q0: np.ndarray) -> np.ndarray:
    from ahrs.filters import Mahony
    f = Mahony(frequency=trial.fs, k_P=0.74, k_I=0.0012)
    q = np.zeros((len(trial.gyr), 4), float)
    q[0] = q0 / np.linalg.norm(q0)
    for k in range(1, len(q)):
        if mode == "9d":
            q[k] = f.updateMARG(q[k-1], trial.gyr[k], trial.acc[k], trial.mag[k])
        else:
            q[k] = f.updateIMU(q[k-1], trial.gyr[k], trial.acc[k])
    return q


def run_vqf(trial: TrialData, mode: str, q_gt: np.ndarray) -> np.ndarray:
    from vqf import VQF
    f = VQF(gyrTs=1.0/trial.fs)
    if mode == "9d":
        out = f.updateBatch(np.ascontiguousarray(trial.gyr), np.ascontiguousarray(trial.acc), np.ascontiguousarray(trial.mag))
        q = np.asarray(out["quat9D"], float)
    else:
        out = f.updateBatch(np.ascontiguousarray(trial.gyr), np.ascontiguousarray(trial.acc))
        q = np.asarray(out["quat6D"], float)
    return align_quat_sequence_first(q, q_gt)


def method_predictions_final(trial, q_gt, convention, mode, ai_models, seq):
    preds = {
        "Gyro-only": run_gyro_only(trial.gyr, q_gt[0], trial.fs, convention.integration_variant),
        "Madgwick": run_madgwick(trial, mode, q_gt[0]),
        "Mahony": run_mahony(trial, mode, q_gt[0]),
        "VQF": run_vqf(trial, mode, q_gt),
    }
    reliability, timing = [], {}
    params = MekfParams()
    fixed = run_mekf(
        trial.gyr, trial.acc, q_gt[0], trial.fs, convention.gravity_global,
        convention.integration_variant,
        trial.mag if mode == "9d" else None,
        convention.magnetic_global if mode == "9d" else None,
        params=params,
    )
    preds["Fixed-MEKF"] = fixed["quat"]
    acc_sig, mag_sig = adaptive_sigmas(trial.acc, trial.mag if mode == "9d" else None)
    adaptive = run_mekf(
        trial.gyr, trial.acc, q_gt[0], trial.fs, convention.gravity_global,
        convention.integration_variant,
        trial.mag if mode == "9d" else None,
        convention.magnetic_global if mode == "9d" else None,
        acc_sig, mag_sig, params,
    )
    preds["Adaptive-MEKF"] = adaptive["quat"]

    raw_g = normalize_rows(trial.acc)
    raw_m = normalize_rows(trial.mag) if mode == "9d" else None
    for family, paths in ai_models.items():
        out, diag = ensemble_predict(paths, seq.features)
        timing[f"{family}_{mode}"] = diag
        if family == "TrustNet":
            g_obs, m_obs = raw_g, raw_m
            g_sigma, m_sigma = out["g_sigma"], out.get("m_sigma")
        elif family == "RefNet":
            g_obs, m_obs = out["g_dir"], out.get("m_dir")
            g_sigma = np.full(len(g_obs), np.deg2rad(3.0))
            m_sigma = None if m_obs is None else np.full(len(m_obs), np.deg2rad(5.0))
        elif family == "ProbRefNet":
            g_obs, m_obs = out["g_dir"], out.get("m_dir")
            g_sigma, m_sigma = out["g_sigma"], out.get("m_sigma")
        else:
            raise ValueError(family)
        result = run_mekf(
            trial.gyr, g_obs, q_gt[0], trial.fs, convention.gravity_global,
            convention.integration_variant,
            m_obs if mode == "9d" else None,
            convention.magnetic_global if mode == "9d" else None,
            g_sigma, m_sigma, params,
        )
        preds[family + "-MEKF"] = result["quat"]
        if family in ("TrustNet", "ProbRefNet"):
            actual_g = seq.raw_gravity_error if family == "TrustNet" else vector_angle(g_obs, seq.gravity_target)
            reliability.append({"trial": trial.name, "mode": mode, "family": family, **reliability_metrics(g_sigma, actual_g, "gravity")})
            if mode == "9d" and m_sigma is not None:
                actual_m = seq.raw_magnetic_error if family == "TrustNet" else vector_angle(m_obs, seq.magnetic_target)
                reliability.append({"trial": trial.name, "mode": mode, "family": family, **reliability_metrics(m_sigma, actual_m, "magnetic")})

    oracle = run_mekf(
        trial.gyr, seq.gravity_target, q_gt[0], trial.fs, convention.gravity_global,
        convention.integration_variant,
        seq.magnetic_target if mode == "9d" else None,
        convention.magnetic_global if mode == "9d" else None,
        np.full(len(trial.gyr), np.deg2rad(1.0)),
        np.full(len(trial.gyr), np.deg2rad(1.0)) if mode == "9d" else None,
        params,
    )
    preds["Oracle-reference-MEKF"] = oracle["quat"]
    return preds, reliability, timing


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    use = df[cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(use[c]):
            use[c] = use[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    return use.to_markdown(index=False)


def write_report(summary, convention, leaderboard, reliability, training, out_path):
    six = leaderboard[(leaderboard["mode"] == "6d") & (leaderboard["method"] != "Oracle-reference-MEKF")].sort_values("inclination_rmse_deg")
    nine = leaderboard[(leaderboard["mode"] == "9d") & (leaderboard["method"] != "Oracle-reference-MEKF")].sort_values("total_rmse_deg")
    lines = ["# Public-data AI-AHRS benchmark", "", "## Key findings", ""]
    if len(six):
        r = six.iloc[0]
        lines.append(f"- 6-axis held-out winner: **{r['method']}**, inclination RMSE **{r['inclination_rmse_deg']:.3f}°**.")
    if len(nine):
        r = nine.iloc[0]
        lines.append(f"- 9-axis held-out winner: **{r['method']}**, total RMSE **{r['total_rmse_deg']:.3f}°**, heading RMSE **{r['heading_rmse_deg']:.3f}°**.")
    lines += [
        "- Six-axis absolute yaw is not observable, so six-axis methods are ranked by inclination and relative-rotation metrics.",
        "- All filters start at the first finite optical GT pose; later optical-GT gaps are excluded from supervision/metrics without deleting IMU time samples.",
        "",
        "## Protocol", "",
        f"- BROAD: {summary['trial_count']} trials, {summary['total_seconds']/60:.1f} min raw data.",
        f"- Split: {len(summary['splits']['train'])} train / {len(summary['splits']['val'])} val / {len(summary['splits']['test'])} held-out test trajectories.",
        f"- Inferred GT convention: body-to-global={convention.body_to_global}; gyro update={convention.integration_variant}.",
        "- Screening uses one neural seed; promising methods should be confirmed with 3+ seeds before publication claims.",
        "",
        "## 6-axis leaderboard", "",
        markdown_table(six, ["method","inclination_rmse_deg","inclination_p95_deg","relative_1s_rmse_deg","relative_10s_rmse_deg","heading_drift_deg_min"]),
        "", "## 9-axis leaderboard", "",
        markdown_table(nine, ["method","total_rmse_deg","heading_rmse_deg","inclination_rmse_deg","total_p95_deg","failure_rate_gt_10deg"]),
        "", "## AI hypothesis test", "",
        "- TrustNet-MEKF learns observation uncertainty only.",
        "- RefNet-MEKF learns cleaned gravity/magnetic directions only.",
        "- ProbRefNet-MEKF jointly learns direction and heteroscedastic uncertainty.",
        "- Oracle-reference-MEKF is a non-deployable upper bound using GT-derived reference directions.",
    ]
    if not reliability.empty:
        rel = reliability.groupby(["mode","family","branch"], as_index=False).agg(
            spearman_sigma_error=("spearman_sigma_error","mean"),
            coverage_1sigma=("coverage_1sigma","mean"),
            coverage_2sigma=("coverage_2sigma","mean"),
            mean_sigma_deg=("mean_sigma_deg","mean"),
            mean_actual_error_deg=("mean_actual_error_deg","mean"),
        )
        lines += ["", "## Uncertainty diagnostics", "", markdown_table(rel, list(rel.columns))]
    lines += ["", "## Training", "", markdown_table(training, ["mode","family","seed","best_val","parameter_count","epochs_run"])]
    lines += ["", "## Interpretation boundary", "", "Arbitrary linear acceleration and arbitrary magnetic disturbance are not identifiable from 9-axis measurements alone. Learning can exploit temporal priors and calibrated uncertainty, but cannot create missing physical information."]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, default=Path(".benchmark_work"))
    p.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--target-hz", type=float, default=50.0)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-train-windows", type=int, default=30000)
    p.add_argument("--max-val-windows", type=int, default=10000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    root = clone_broad(args.work_dir.resolve()/"broad")
    raw_trials, _ = load_all_trials(root)
    raw_summary = summarize_trials(raw_trials)
    trials = {n: downsample_trial(t, args.target_hz) for n,t in raw_trials.items()}
    convention = infer_convention({n:t for n,t in trials.items() if get_split_name(n)=="train"})
    seq_by_mode = {m: prepare_sequences(trials, convention, m) for m in ("6d","9d")}
    lookup = {m:{s.name:s for s in seqs} for m,seqs in seq_by_mode.items()}
    cfg = TrainConfig(window=args.window, epochs=args.epochs, max_train_windows=args.max_train_windows, max_val_windows=args.max_val_windows)
    model_map, training_rows = train_ai_models(seq_by_mode, out, args.seeds, cfg)
    training_df = pd.DataFrame(training_rows); training_df.to_csv(out/"training_summary.csv", index=False)

    trial_rows=[]; rel_rows=[]; timing_rows=[]
    test_names=[n for n in trials if get_split_name(n)=="test"]
    for mode in ("6d","9d"):
        print(f"Evaluating FINAL {mode} on {len(test_names)} trajectories")
        for name in test_names:
            tfull=trials[name]
            qfull=canonical_quat(tfull.quat_gt, convention.body_to_global)
            t,q,seq,k0=first_valid_crop(tfull,qfull,lookup[mode][name])
            preds, rel, timing = method_predictions_final(t,q,convention,mode,model_map[mode],seq)
            rel_rows += rel
            timing_rows += [{"trial":name,"mode":mode,"method":k,**v} for k,v in timing.items()]
            for method,qhat in preds.items():
                trial_rows.append({"trial":name,"mode":mode,"method":method,"crop_start":k0,**orientation_metrics(qhat,q,t.movement,t.fs)})

    pd.DataFrame(trial_rows).to_csv(out/"test_trial_metrics.csv", index=False)
    leaderboard=pd.DataFrame(aggregate_by_method(trial_rows,"6d")+aggregate_by_method(trial_rows,"9d")); leaderboard.to_csv(out/"leaderboard.csv",index=False)
    group_df=pd.DataFrame(aggregate_groups(trial_rows,{n:t.groups for n,t in trials.items()})); group_df.to_csv(out/"group_metrics.csv",index=False)
    rel_df=pd.DataFrame(rel_rows); rel_df.to_csv(out/"uncertainty_metrics.csv",index=False)
    pd.DataFrame(timing_rows).to_csv(out/"runtime_metrics.csv",index=False)
    save_json(out/"protocol.json", {"raw_summary":raw_summary,"downsampled_summary":summarize_trials(trials),"target_hz":args.target_hz,"window":args.window,"seeds":args.seeds,"convention":{"body_to_global":convention.body_to_global,"integration_variant":convention.integration_variant,"gravity_global":convention.gravity_global.tolist(),"magnetic_global":convention.magnetic_global.tolist()}})
    save_json(out/"errors.json", [])
    save_json(out/"environment.json", {"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__})
    make_plots(leaderboard,group_df,out/"plots")
    write_report(raw_summary,convention,leaderboard,rel_df,training_df,out/"REPORT.md")
    print("FINAL LEADERBOARD")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
