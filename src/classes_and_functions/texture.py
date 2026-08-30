"""ODFs, symmetry, empirical texture estimation, and JSON-friendly ODF input.

``evaluate`` returns density relative to normalized Haar measure on SO(3),
whose integral is one.  Crystal symmetry is deliberately opt-in: sampling
keeps explicit grain orientations; symmetry utilities are used when comparing
or displaying equivalent orientations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import itertools
import json
import numpy as np

from .orientation import (axis_angle_to_quaternion, canonicalize_quaternion,
    compose_rotations, quaternion_distance, quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion, uniform_quaternions)


def _rng(rng): return np.random.default_rng() if rng is None else rng
def _unit(v):
    v = np.asarray(v, dtype=float); n = np.linalg.norm(v)
    if n == 0: raise ValueError("Direction must be nonzero.")
    return v / n

@lru_cache(maxsize=128)
def _component_grid(spread: float):
    theta = np.linspace(0, np.pi, 4097)
    # Haar angle density on SO(3); its integral is 1.
    density = (2/np.pi) * np.sin(theta/2)**2 * np.exp(-theta**2/(2*spread**2))
    z = np.trapezoid(density, theta)
    cdf = np.concatenate(([0.], np.cumsum((density[1:]+density[:-1])*np.diff(theta)/2))) / z
    return theta, cdf, z

class ODF(ABC):
    @abstractmethod
    def evaluate(self, orientation: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray: ...

class RandomODF(ODF):
    """Haar-uniform, isotropic ODF (density exactly one)."""
    def evaluate(self, orientation): return np.ones(np.asarray(orientation).shape[:-1])
    def sample(self, n, rng=None): return uniform_quaternions(n, _rng(rng))


class RandomOrientationSampler:
    """Explicit reproducible orientation-generation interface for isotropy."""
    def __init__(self, seed=None): self.rng = np.random.default_rng(seed)
    def sample(self, n): return uniform_quaternions(n, self.rng)

class OrientationComponent(ODF):
    """Isotropic Gaussian-in-geodesic-angle component centered at ``center``.

    Its density is ``exp(-d(q,q0)^2/(2 s^2))/Z(s)`` relative to Haar measure;
    Z is evaluated by one-dimensional quadrature and sampling uses its inverse
    CDF, so it remains normalized at broad spreads.
    """
    def __init__(self, center=(1, 0, 0, 0), spread_deg=5.0):
        self.center = canonicalize_quaternion(np.asarray(center, float))
        self.spread_deg = float(spread_deg)
        self.spread = np.deg2rad(self.spread_deg)
        if self.spread <= 0: raise ValueError("spread_deg must be positive.")
    def evaluate(self, orientation):
        _, _, z = _component_grid(self.spread)
        d = quaternion_distance(orientation, self.center)
        return np.exp(-d*d/(2*self.spread*self.spread)) / z
    def sample(self, n, rng=None):
        rng = _rng(rng); theta, cdf, _ = _component_grid(self.spread)
        angle = np.interp(rng.random(n), cdf, theta)
        axis = rng.normal(size=(n, 3)); axis /= np.linalg.norm(axis, axis=1)[:, None]
        return canonicalize_quaternion(compose_rotations(np.broadcast_to(self.center, (n, 4)), axis_angle_to_quaternion(axis, angle)))


class SharpOrientationODF(OrientationComponent):
    """Numerically finite delta-like texture; default one-degree spread."""
    def __init__(self, center=(1, 0, 0, 0), spread_deg=1.0):
        super().__init__(center=center, spread_deg=spread_deg)

def _rotation_between(a, b):
    a, b = _unit(a), _unit(b); dot = np.clip(a @ b, -1, 1)
    if dot < -1 + 1e-12:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-12: axis = np.cross(a, [0, 1, 0])
        return axis_angle_to_quaternion(axis, np.pi)
    return axis_angle_to_quaternion(np.cross(a, b) if dot < 1 else np.array([1., 0, 0]), np.arccos(dot))

@lru_cache(maxsize=128)
def _fiber_grid(spread):
    beta = np.linspace(0, np.pi, 4097)
    p = .5*np.sin(beta)*np.exp(-beta**2/(2*spread**2))
    z = np.trapezoid(p, beta)
    cdf = np.concatenate(([0.], np.cumsum((p[1:]+p[:-1])*np.diff(beta)/2))) / z
    return beta, cdf, z

class FiberODF(ODF):
    """Fiber with a finite Gaussian angular spread.

    ``R @ crystal_direction`` is distributed around ``sample_direction``;
    rotation around that sample axis is uniform.  The density is normalized
    relative to Haar measure and depends only on that misalignment angle.
    """
    def __init__(self, crystal_direction, sample_direction=(0, 0, 1), spread_deg=10):
        self.crystal_direction, self.sample_direction = _unit(crystal_direction), _unit(sample_direction)
        self.spread_deg, self.spread = float(spread_deg), np.deg2rad(float(spread_deg))
        if self.spread <= 0: raise ValueError("spread_deg must be positive.")
    def evaluate(self, orientation):
        q = np.asarray(orientation); direction = np.einsum("...ij,j->...i", quaternion_to_rotation_matrix(q), self.crystal_direction)
        beta = np.arccos(np.clip(direction @ self.sample_direction, -1, 1)); _, _, z = _fiber_grid(self.spread)
        return np.exp(-beta*beta/(2*self.spread*self.spread)) / z
    def sample(self, n, rng=None):
        rng = _rng(rng); beta_grid, cdf, _ = _fiber_grid(self.spread)
        beta, azimuth, spin = np.interp(rng.random(n), cdf, beta_grid), rng.uniform(0, 2*np.pi, n), rng.uniform(0, 2*np.pi, n)
        # Orthonormal basis perpendicular to sample direction.
        temp = np.array([1., 0, 0]) if abs(self.sample_direction[0]) < .9 else np.array([0., 1, 0])
        e1 = _unit(np.cross(self.sample_direction, temp)); e2 = np.cross(self.sample_direction, e1)
        targets = (np.cos(beta)[:, None]*self.sample_direction + np.sin(beta)[:, None]*(np.cos(azimuth)[:, None]*e1 + np.sin(azimuth)[:, None]*e2))
        base = np.array([_rotation_between(self.crystal_direction, t) for t in targets])
        spin_q = axis_angle_to_quaternion(targets, spin)
        return canonicalize_quaternion(compose_rotations(spin_q, base))

class MixtureODF(ODF):
    def __init__(self, components):
        self.weights = np.asarray([x[0] for x in components], float)
        self.components = [x[1] for x in components]
        if len(self.components) == 0 or np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError("Mixture weights must be nonnegative and sum to one.")
    def evaluate(self, orientation): return sum(w*c.evaluate(orientation) for w, c in zip(self.weights, self.components))
    def sample(self, n, rng=None):
        rng = _rng(rng); choices = rng.choice(len(self.components), n, p=self.weights); result = np.empty((n, 4))
        for index, component in enumerate(self.components):
            mask = choices == index; result[mask] = component.sample(mask.sum(), rng)
        return result

class EmpiricalODF(ODF):
    """Volume-weighted kernel estimate from explicit grain quaternions."""
    def __init__(self, orientations, weights=None, spread_deg=10):
        self.orientations = canonicalize_quaternion(np.asarray(orientations, float))
        self.weights = np.ones(len(self.orientations)) if weights is None else np.asarray(weights, float)
        if len(self.weights) != len(self.orientations) or np.any(self.weights < 0) or self.weights.sum() == 0: raise ValueError("Invalid empirical ODF weights.")
        self.weights /= self.weights.sum(); self.kernel = OrientationComponent(spread_deg=spread_deg)
    def evaluate(self, orientation):
        q = np.asarray(orientation); d = quaternion_distance(q[..., None, :], self.orientations)
        _, _, z = _component_grid(self.kernel.spread)
        return np.sum(np.exp(-d*d/(2*self.kernel.spread**2)) * self.weights, axis=-1) / z
    def sample(self, n, rng=None):
        rng = _rng(rng); selected = rng.choice(len(self.orientations), n, p=self.weights)
        return np.vstack([OrientationComponent(self.orientations[i], self.kernel.spread_deg).sample(1, rng) for i in selected])

def estimate_odf(microstructure, weights=None, spread_deg=10):
    """Estimate ODF from an iterable of grains or a quaternion ``(n,4)`` array.

    Grain ``volume`` is used automatically when present; otherwise all grains
    have equal weight.
    """
    if isinstance(microstructure, np.ndarray): return EmpiricalODF(microstructure, weights, spread_deg)
    grains = list(getattr(microstructure, "grains", microstructure))
    orientations = np.array([g.orientation for g in grains])
    if weights is None: weights = [getattr(g, "volume", 1.0) for g in grains]
    return EmpiricalODF(orientations, weights, spread_deg)

def get_symmetry_operations(crystal):
    crystal = str(crystal).lower()
    if crystal in {"cubic", "fcc", "bcc", "sc"}:
        matrices = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                m = np.zeros((3, 3)); m[range(3), perm] = signs
                if round(np.linalg.det(m)) == 1: matrices.append(m)
        return rotation_matrix_to_quaternion(np.array(matrices))
    if crystal in {"hexagonal", "hcp"}:
        ops = []
        for k in range(6):
            angle = k*np.pi/3; ops.append(axis_angle_to_quaternion([0, 0, 1], angle))
            axis = [np.cos(angle/2), np.sin(angle/2), 0]; ops.append(axis_angle_to_quaternion(axis, np.pi))
        return canonicalize_quaternion(np.array(ops))
    if crystal in {"triclinic", "none", "identity"}: return np.array([[1., 0, 0, 0]])
    raise ValueError(f"Unsupported crystal symmetry: {crystal}")

def equivalent_orientations(q, crystal):
    q = canonicalize_quaternion(q); ops = get_symmetry_operations(crystal)
    return canonicalize_quaternion(compose_rotations(q, ops))

def reduce_to_fundamental_zone(q, crystal):
    candidates = equivalent_orientations(q, crystal)
    # A deterministic canonical representative; this is not a geometric FZ chart.
    return candidates[np.lexsort(candidates[:, ::-1].T)[-1]]

def odf_from_dict(config):
    """Build an ODF from a JSON/YAML-decoded mapping for future file import."""
    kind = config["type"].lower()
    if kind in {"random", "isotropic"}: return RandomODF()
    if kind == "sharp": return SharpOrientationODF(config.get("center_quaternion", [1,0,0,0]), config.get("spread_deg", 1))
    if kind in {"component", "gaussian"}: return OrientationComponent(config.get("center_quaternion", [1,0,0,0]), config.get("spread_deg", 5))
    if kind == "fiber": return FiberODF(config["crystal_direction"], config.get("sample_direction", [0,0,1]), config.get("spread_deg", 10))
    if kind == "mixture": return MixtureODF([(item["weight"], odf_from_dict(item)) for item in config["components"]])
    raise ValueError(f"Unknown ODF type {kind!r}")

def odf_from_json(path):
    with open(path) as handle: return odf_from_dict(json.load(handle)["odf"])
