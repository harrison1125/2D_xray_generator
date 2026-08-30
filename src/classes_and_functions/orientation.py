"""Orientation primitives used by the diffraction simulator.

Convention
----------
Quaternions are stored as ``[w, x, y, z]`` and are unit quaternions.  They
represent *active*, right-handed rotations from crystal coordinates into sample
coordinates: ``v_sample = R(q) @ v_crystal``.  Reciprocal vectors follow the
same convention.  Euler compatibility functions use active, intrinsic Bunge
ZXZ angles ``(phi1, Phi, phi2)`` in radians, so
``R = Rz(phi1) @ Rx(Phi) @ Rz(phi2)`` for column vectors.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Return normalized ``wxyz`` quaternion(s); reject zero quaternions."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("A zero quaternion has no orientation.")
    return q / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert normalized ``wxyz`` quaternion(s) to active rotation matrix(es)."""
    q = quaternion_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w),
        2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert active rotation matrix/matrices to canonical-sign ``wxyz``."""
    matrix = np.asarray(matrix, dtype=float)
    xyzw = Rotation.from_matrix(matrix.reshape((-1, 3, 3))).as_quat()
    q = xyzw[:, [3, 0, 1, 2]].reshape(matrix.shape[:-2] + (4,))
    return canonicalize_quaternion(q)


def euler_to_quaternion(phi1: np.ndarray, Phi: np.ndarray | None = None,
                        phi2: np.ndarray | None = None, *, degrees: bool = False) -> np.ndarray:
    """Convert Bunge intrinsic-ZXZ angles to ``wxyz`` quaternion(s)."""
    angles = np.asarray(phi1 if Phi is None and phi2 is None else np.stack((phi1, Phi, phi2), axis=-1), dtype=float)
    if angles.shape[-1] != 3:
        raise ValueError("Euler angles must have final dimension 3: (phi1, Phi, phi2).")
    xyzw = Rotation.from_euler("ZXZ", angles.reshape((-1, 3)), degrees=degrees).as_quat()
    return canonicalize_quaternion(xyzw[:, [3, 0, 1, 2]].reshape(angles.shape[:-1] + (4,)))


def quaternion_to_euler(q: np.ndarray, *, degrees: bool = False) -> np.ndarray:
    """Convert ``wxyz`` to active intrinsic Bunge-ZXZ Euler angles."""
    q = quaternion_normalize(q)
    xyzw = q[..., [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw.reshape((-1, 4))).as_euler("ZXZ", degrees=degrees).reshape(q.shape[:-1] + (3,))


def compose_rotations(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return ``q1 ∘ q2``: apply q2 then q1 (matrix product R1 @ R2)."""
    q1, q2 = np.broadcast_arrays(quaternion_normalize(q1), quaternion_normalize(q2))
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return quaternion_normalize(np.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ), axis=-1))


def rotate_vector(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Actively rotate vector(s) from crystal into sample coordinates."""
    return np.einsum("...ij,...j->...i", quaternion_to_rotation_matrix(q), np.asarray(vector, dtype=float))


def canonicalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize and choose q/-q deterministically (nonnegative scalar part)."""
    q = quaternion_normalize(q)
    sign = np.where(q[..., :1] < 0, -1.0, 1.0)
    return q * sign


def uniform_quaternions(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample Haar-uniform orientations on SO(3) using Shoemake's algorithm."""
    rng = np.random.default_rng() if rng is None else rng
    u1, u2, u3 = rng.random((3, n))
    q = np.column_stack((
        np.sqrt(u1) * np.cos(2*np.pi*u3),
        np.sqrt(1-u1) * np.sin(2*np.pi*u2),
        np.sqrt(1-u1) * np.cos(2*np.pi*u2),
        np.sqrt(u1) * np.sin(2*np.pi*u3),
    ))
    # The construction above is wxyz, not scipy's xyzw.
    return canonicalize_quaternion(q)


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic distance in [0, pi] on SO(3), identifying q and -q."""
    q1, q2 = np.broadcast_arrays(quaternion_normalize(q1), quaternion_normalize(q2))
    dot = np.clip(np.abs(np.sum(q1*q2, axis=-1)), 0.0, 1.0)
    return 2 * np.arccos(dot)


def axis_angle_to_quaternion(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.asarray(angle, dtype=float)
    return canonicalize_quaternion(np.concatenate((np.cos(angle[..., None]/2), axis*np.sin(angle[..., None]/2)), axis=-1))
