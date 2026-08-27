from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize_rows(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return normalize_rows(q)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return quat_normalize(out)


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1e-8
    scale[small] = 0.5 - (angle[small] ** 2) / 48.0
    scale[~small] = np.sin(half[~small]) / angle[~small]
    q = np.concatenate([np.cos(half), rv * scale], axis=-1)
    return quat_normalize(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    w, x, y, z = np.moveaxis(q, -1, 0)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


def quat_angle(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return 2.0 * np.arccos(np.clip(np.abs(q[..., 0]), 0.0, 1.0))


def quat_error(q_est: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    return quat_mul(q_est, quat_conj(q_gt))


def align_quat_sequence_first(q_est: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    q_est = quat_normalize(q_est)
    q_gt = quat_normalize(q_gt)
    q_align = quat_mul(q_gt[0], quat_conj(q_est[0]))
    return quat_mul(np.broadcast_to(q_align, q_est.shape), q_est)


def calculate_total_error(q_est: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    return quat_angle(quat_error(q_est, q_gt))


def calculate_heading_inclination_error(q_est: np.ndarray, q_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qe = quat_error(q_est, q_gt)
    w = qe[:, 0]
    z = qe[:, 3]
    heading = 2.0 * np.arctan2(np.abs(z), np.clip(np.abs(w), EPS, None))
    inclination = 2.0 * np.arccos(np.clip(np.sqrt(w * w + z * z), 0.0, 1.0))
    return heading, inclination


def signed_heading_error(q_est: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    qe = quat_error(q_est, q_gt)
    return 2.0 * np.arctan2(qe[:, 3], qe[:, 0])


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotate_body_to_global(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q)
    return np.einsum("...ij,...j->...i", R, v_body)


def rotate_global_to_body(q: np.ndarray, v_global: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q)
    return np.einsum("...ji,...j->...i", R, v_global)


def vector_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize_rows(a)
    b = normalize_rows(b)
    dot = np.sum(a * b, axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def ensure_quat_continuity(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q).copy()
    for k in range(1, len(q)):
        if np.dot(q[k - 1], q[k]) < 0:
            q[k] *= -1.0
    return q
