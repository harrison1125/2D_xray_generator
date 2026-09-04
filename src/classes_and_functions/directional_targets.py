"""Pole-figure and inverse-pole-figure targets from realized orientations.

The simulator stores active crystal-to-sample quaternions.  This module turns
the exact finite orientation realization used for one diffraction pattern into
symmetry-complete, antipodal pole distributions on a normalized Lambert
equal-area upper-hemisphere disk.  It also stores a compact 2-D DCT basis for
direct use by the Bayesian pole/IPF branch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import dctn

from .orientation import quaternion_to_rotation_matrix
from .texture import get_symmetry_operations


def _unit_vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("Directions must be finite three-vectors.")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Directions must be nonzero.")
    return vector / norm


def direct_lattice_basis(unit_cell: dict[str, float]) -> np.ndarray:
    """Return direct-lattice vectors as columns in a Cartesian crystal frame."""
    required = {"a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg"}
    if not isinstance(unit_cell, dict) or set(unit_cell) != required:
        raise ValueError(f"unit_cell must contain exactly {sorted(required)}.")
    a, b, c = (float(unit_cell[name]) for name in ("a", "b", "c"))
    alpha, beta, gamma = np.deg2rad([
        unit_cell["alpha_deg"], unit_cell["beta_deg"], unit_cell["gamma_deg"]
    ])
    sin_gamma = float(np.sin(gamma))
    if min(a, b, c) <= 0 or abs(sin_gamma) < 1e-12:
        raise ValueError("unit_cell must have positive lengths and a nonsingular gamma angle.")
    third = np.array([
        c * np.cos(beta),
        c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma,
        0.0,
    ])
    third[2] = np.sqrt(max(0.0, c * c - third[0] ** 2 - third[1] ** 2))
    basis = np.column_stack((
        [a, 0.0, 0.0],
        [b * np.cos(gamma), b * sin_gamma, 0.0],
        third,
    ))
    if abs(float(np.linalg.det(basis))) < 1e-14:
        raise ValueError("unit_cell does not form a positive-volume lattice basis.")
    return basis


def reciprocal_plane_normal(hkl, unit_cell: dict[str, float] | None = None) -> np.ndarray:
    """Return a unit Cartesian normal for a three-index reciprocal plane."""
    indices = np.asarray(hkl, dtype=float)
    if indices.shape != (3,) or not np.all(np.isfinite(indices)) or not np.any(indices):
        raise ValueError("hkl must be a finite, nonzero three-vector.")
    if unit_cell is None:
        return _unit_vector(indices)
    # With direct vectors as columns, reciprocal vectors (without the
    # irrelevant 2*pi factor) are columns of inv(A).T.
    reciprocal_basis = np.linalg.inv(direct_lattice_basis(unit_cell)).T
    return _unit_vector(reciprocal_basis @ indices)


def _unique_antipodal_axes(vectors: np.ndarray, decimals: int = 12) -> np.ndarray:
    axes = np.asarray(vectors, dtype=float).reshape(-1, 3)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    # Choose a deterministic representative for each v/-v pair before
    # deduplicating a symmetry orbit.
    for index, vector in enumerate(axes):
        first = np.flatnonzero(np.abs(vector) > 10 ** (-decimals))
        if first.size and vector[first[0]] < 0:
            axes[index] *= -1
    rounded = np.round(axes, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return axes[np.sort(unique_indices)]


def symmetry_equivalent_axes(direction, crystal_symmetry: str) -> np.ndarray:
    """Return unique antipodal axes in one proper crystal-symmetry orbit."""
    matrices = quaternion_to_rotation_matrix(get_symmetry_operations(crystal_symmetry))
    orbit = np.einsum("hij,j->hi", matrices, _unit_vector(direction))
    return _unique_antipodal_axes(orbit)


def lambert_equal_area_xy(vectors: np.ndarray) -> np.ndarray:
    """Project unoriented axes onto a unit-radius upper-hemisphere disk."""
    directions = np.asarray(vectors, dtype=float).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions.copy()
    directions[directions[:, 2] < 0] *= -1
    denominator = np.sqrt(np.clip(1.0 + directions[:, 2], 1e-15, None))
    return directions[:, :2] / denominator[:, None]


def lambert_probability_mass(vectors: np.ndarray, grid_size: int) -> np.ndarray:
    """Histogram axes as probability mass on an equal-area Lambert disk."""
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    xy = lambert_equal_area_xy(vectors)
    histogram, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=grid_size, range=((-1.0, 1.0), (-1.0, 1.0))
    )
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("No directions landed on the Lambert projection grid.")
    return histogram / total


def lambert_disk_mask(grid_size: int) -> np.ndarray:
    centers = -1.0 + (np.arange(grid_size, dtype=float) + 0.5) * 2.0 / grid_size
    x, y = np.meshgrid(centers, centers)
    return x * x + y * y <= 1.0


def _compact_coefficients(maps: np.ndarray, coefficient_shape: tuple[int, int]) -> np.ndarray:
    rows, columns = map(int, coefficient_shape)
    if rows < 1 or columns < 1 or rows > maps.shape[-2] or columns > maps.shape[-1]:
        raise ValueError("coefficient_shape must be positive and no larger than the map grid.")
    transformed = dctn(maps, axes=(-2, -1), norm="ortho")
    return transformed[..., :rows, :columns].reshape(len(maps), rows * columns)


def build_directional_targets(
    orientations: np.ndarray,
    *,
    crystal_symmetry: str,
    pole_families: list[dict],
    sample_directions: list[dict],
    grid_size: int = 32,
    coefficient_shape: tuple[int, int] = (9, 9),
    unit_cell: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build PF/IPF maps and compact coefficients for one observation."""
    quaternions = np.asarray(orientations, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or len(quaternions) < 1:
        raise ValueError("orientations must have shape (n, 4) with n > 0.")
    if not pole_families or not sample_directions:
        raise ValueError("At least one pole family and sample direction are required.")
    rotations = quaternion_to_rotation_matrix(quaternions)
    symmetry_matrices = quaternion_to_rotation_matrix(
        get_symmetry_operations(crystal_symmetry)
    )

    pole_maps, pole_names, pole_normals, pole_hkl = [], [], [], []
    for family in pole_families:
        normal = reciprocal_plane_normal(family["hkl"], unit_cell)
        orbit = symmetry_equivalent_axes(normal, crystal_symmetry)
        sample_axes = np.einsum("nij,hj->nhi", rotations, orbit).reshape(-1, 3)
        pole_maps.append(lambert_probability_mass(sample_axes, grid_size))
        pole_names.append(str(family["name"]))
        pole_normals.append(normal)
        pole_hkl.append(np.asarray(family["hkl"], dtype=np.int16))

    ipf_maps, sample_names, normalized_sample_directions = [], [], []
    for item in sample_directions:
        direction = _unit_vector(item["direction"])
        crystal_axes = np.einsum("nji,j->ni", rotations, direction)
        equivalent = np.einsum(
            "hji,nj->nhi", symmetry_matrices, crystal_axes
        ).reshape(-1, 3)
        ipf_maps.append(lambert_probability_mass(equivalent, grid_size))
        sample_names.append(str(item["name"]))
        normalized_sample_directions.append(direction)

    pole_maps_array = np.asarray(pole_maps, dtype=np.float32)
    ipf_maps_array = np.asarray(ipf_maps, dtype=np.float32)
    all_maps = np.concatenate((pole_maps_array, ipf_maps_array), axis=0)
    coefficients = _compact_coefficients(all_maps, coefficient_shape).astype(np.float32)
    directional_names = [f"PF {name}" for name in pole_names]
    directional_names.extend(f"IPF {name}" for name in sample_names)
    return {
        "target_version": np.asarray("realized_pf_ipf_lambert_dct_v1"),
        "target_source": np.asarray("finite_orientation_realization_used_by_diffraction_pattern"),
        "projection": np.asarray("normalized_lambert_equal_area_upper_hemisphere"),
        "crystal_symmetry": np.asarray(crystal_symmetry),
        "orientation_count": np.asarray(len(quaternions), dtype=np.int32),
        "grid_size": np.asarray(grid_size, dtype=np.int16),
        "coefficient_shape": np.asarray(coefficient_shape, dtype=np.int16),
        "pole_figure_probability_mass": pole_maps_array,
        "inverse_pole_figure_probability_mass": ipf_maps_array,
        "directional_coefficients": coefficients,
        "directional_mask": np.ones(len(all_maps), dtype=bool),
        "directional_names": np.asarray(directional_names),
        "pole_names": np.asarray(pole_names),
        "pole_hkl_3_index": np.asarray(pole_hkl, dtype=np.int16),
        "pole_crystal_cartesian_normals": np.asarray(pole_normals, dtype=np.float64),
        "sample_direction_names": np.asarray(sample_names),
        "sample_directions": np.asarray(normalized_sample_directions, dtype=np.float64),
        "valid_disk_mask": lambert_disk_mask(grid_size),
    }


def write_directional_targets(path: str | Path, *args, **kwargs) -> None:
    """Write :func:`build_directional_targets` output as a compressed NPZ."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **build_directional_targets(*args, **kwargs))
