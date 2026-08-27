from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import EPS, quat_from_rotvec, quat_mul, quat_normalize, quat_to_rotmat, skew


@dataclass
class MekfParams:
    gyro_noise: float = np.deg2rad(0.12)
    bias_rw: float = np.deg2rad(0.01)
    initial_att_sigma: float = np.deg2rad(3.0)
    initial_bias_sigma: float = np.deg2rad(1.0)
    acc_sigma: float = np.deg2rad(5.0)
    mag_sigma: float = np.deg2rad(8.0)
    min_sigma: float = np.deg2rad(0.5)
    max_sigma: float = np.deg2rad(89.0)
    nis_gate: float = 16.27
    estimate_bias: bool = True


def _propagate_quat(q: np.ndarray, omega: np.ndarray, dt: float, variant: str) -> np.ndarray:
    sign = -1.0 if variant.endswith("-") else 1.0
    dq = quat_from_rotvec(sign * omega * dt)
    if variant.startswith("left"):
        return quat_mul(dq, q)
    return quat_mul(q, dq)


def _predict_body_reference(q_bg: np.ndarray, ref_global: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q_bg)
    return R.T @ ref_global


def _direction_update(
    q: np.ndarray,
    bias: np.ndarray,
    P: np.ndarray,
    z_body: np.ndarray,
    ref_global: np.ndarray,
    sigma: float,
    nis_gate: float,
    estimate_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
    if not np.all(np.isfinite(z_body)) or np.linalg.norm(z_body) < EPS or not np.isfinite(sigma):
        return q, bias, P, False, np.nan
    sigma = float(max(sigma, 1e-6))
    z = z_body / np.linalg.norm(z_body)
    h = _predict_body_reference(q, ref_global)
    h = h / max(np.linalg.norm(h), EPS)
    residual = z - h
    H = np.zeros((3, 6), dtype=np.float64)
    H[:, :3] = skew(h)
    if not estimate_bias:
        H = H[:, :3]
        P_use = P[:3, :3]
    else:
        P_use = P
    Rm = (sigma * sigma) * np.eye(3)
    S = H @ P_use @ H.T + Rm
    try:
        Sinv_res = np.linalg.solve(S, residual)
        nis = float(residual @ Sinv_res)
    except np.linalg.LinAlgError:
        return q, bias, P, False, np.nan
    if nis > nis_gate:
        inflate = min(100.0, max(1.0, nis / nis_gate))
        Rm *= inflate
        S = H @ P_use @ H.T + Rm
    K = P_use @ H.T @ np.linalg.inv(S)
    dx = K @ residual
    dtheta = dx[:3]
    q = quat_mul(q, quat_from_rotvec(dtheta))
    if estimate_bias:
        bias = bias + dx[3:6]
        I = np.eye(6)
        A = I - K @ H
        P = A @ P @ A.T + K @ Rm @ K.T
    else:
        I3 = np.eye(3)
        A = I3 - K @ H
        P3 = A @ P[:3, :3] @ A.T + K @ Rm @ K.T
        P[:3, :3] = P3
    return quat_normalize(q), bias, P, True, nis


def run_mekf(
    gyr: np.ndarray,
    acc_obs: np.ndarray,
    q0: np.ndarray,
    fs: float,
    gravity_global: np.ndarray,
    integration_variant: str = "right+",
    mag_obs: np.ndarray | None = None,
    magnetic_global: np.ndarray | None = None,
    acc_sigma: np.ndarray | float | None = None,
    mag_sigma: np.ndarray | float | None = None,
    params: MekfParams | None = None,
) -> dict[str, np.ndarray]:
    params = params or MekfParams()
    n = len(gyr)
    if len(acc_obs) != n:
        raise ValueError("gyr and acc length mismatch")
    if mag_obs is not None and len(mag_obs) != n:
        raise ValueError("gyr and mag length mismatch")
    dt = 1.0 / float(fs)
    q_out = np.zeros((n, 4), dtype=np.float64)
    b_out = np.zeros((n, 3), dtype=np.float64)
    nis_acc = np.full(n, np.nan)
    nis_mag = np.full(n, np.nan)
    accepted_acc = np.zeros(n, dtype=bool)
    accepted_mag = np.zeros(n, dtype=bool)

    q = quat_normalize(np.asarray(q0, dtype=np.float64))
    bias = np.zeros(3, dtype=np.float64)
    P = np.diag([params.initial_att_sigma**2] * 3 + [params.initial_bias_sigma**2] * 3)
    acc_sig = np.full(n, params.acc_sigma) if acc_sigma is None else np.broadcast_to(acc_sigma, (n,)).astype(float)
    mag_sig = np.full(n, params.mag_sigma) if mag_sigma is None else np.broadcast_to(mag_sigma, (n,)).astype(float)

    gyro_sign = -1.0 if integration_variant.endswith("-") else 1.0
    for k in range(n):
        if k > 0:
            omega = gyr[k - 1] - bias
            q = _propagate_quat(q, omega, dt, integration_variant)
            F = np.zeros((6, 6), dtype=np.float64)
            F[:3, :3] = -skew(gyro_sign * omega)
            if params.estimate_bias:
                F[:3, 3:6] = -gyro_sign * np.eye(3)
            Phi = np.eye(6) + F * dt
            Qd = np.diag([params.gyro_noise**2] * 3 + [params.bias_rw**2] * 3) * dt
            P = Phi @ P @ Phi.T + Qd

        s_acc = float(np.clip(acc_sig[k], params.min_sigma, params.max_sigma))
        q, bias, P, accepted_acc[k], nis_acc[k] = _direction_update(
            q, bias, P, acc_obs[k], gravity_global, s_acc, params.nis_gate, params.estimate_bias
        )
        if mag_obs is not None and magnetic_global is not None:
            s_mag = float(np.clip(mag_sig[k], params.min_sigma, params.max_sigma))
            q, bias, P, accepted_mag[k], nis_mag[k] = _direction_update(
                q, bias, P, mag_obs[k], magnetic_global, s_mag, params.nis_gate, params.estimate_bias
            )
        q_out[k] = q
        b_out[k] = bias

    return {
        "quat": q_out,
        "bias": b_out,
        "nis_acc": nis_acc,
        "nis_mag": nis_mag,
        "accepted_acc": accepted_acc,
        "accepted_mag": accepted_mag,
    }


def adaptive_sigmas(acc: np.ndarray, mag: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    acc_norm = np.linalg.norm(acc, axis=1)
    a_ref = np.median(acc_norm)
    a_dev = np.abs(acc_norm / max(a_ref, EPS) - 1.0)
    acc_sigma = np.deg2rad(3.0 + 120.0 * np.clip(a_dev, 0.0, 0.7))
    if mag is None:
        return acc_sigma, None
    mag_norm = np.linalg.norm(mag, axis=1)
    m_ref = np.median(mag_norm)
    m_dev = np.abs(mag_norm / max(m_ref, EPS) - 1.0)
    mag_sigma = np.deg2rad(5.0 + 160.0 * np.clip(m_dev, 0.0, 0.5))
    return acc_sigma, mag_sigma


def run_gyro_only(gyr: np.ndarray, q0: np.ndarray, fs: float, integration_variant: str) -> np.ndarray:
    q = quat_normalize(q0)
    out = np.zeros((len(gyr), 4), dtype=np.float64)
    dt = 1.0 / fs
    for k in range(len(gyr)):
        if k > 0:
            q = _propagate_quat(q, gyr[k - 1], dt, integration_variant)
        out[k] = q
    return out
