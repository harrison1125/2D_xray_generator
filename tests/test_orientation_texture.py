import numpy as np
import unittest
from scipy.integrate import quad
from src.classes_and_functions.configuration import validate_config

from src.classes_and_functions.crystal import GrainCubic
from src.classes_and_functions.experiment import Experiment
from src.classes_and_functions.grain_scattering import coherent_length, peak_properties
from src.classes_and_functions.orientation import (
    euler_to_quaternion, quaternion_to_euler, quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion, rotate_vector, uniform_quaternions, compose_rotations,
)
from src.classes_and_functions.texture import (
    FiberODF, MixtureODF, OrientationComponent, PartialFiberODF, RandomODF, estimate_odf,
    SharpOrientationODF, get_symmetry_operations, odf_from_dict,
)
from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment
from src.classes_and_functions.continuous_odf import (
    SobolODFSpace, dvp_exponent_from_fwhm, dvp_log_normalization,
    harmonic_coefficients, sobol_unit_cube,
)


def test_quaternion_identity_and_ninety_degree_rotation():
    assert np.allclose(rotate_vector([1, 0, 0, 0], [1, 2, 3]), [1, 2, 3])
    q = euler_to_quaternion([np.pi / 2, 0, 0])
    assert np.allclose(rotate_vector(q, [1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_rotation_and_euler_round_trips():
    q = uniform_quaternions(20, np.random.default_rng(8))
    reconstructed = rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(q))
    assert np.allclose(np.abs(np.sum(q * reconstructed, axis=1)), 1, atol=1e-12)
    angles = np.array([.3, .7, 1.2])
    q2 = euler_to_quaternion(quaternion_to_euler(euler_to_quaternion(angles)))
    assert np.isclose(abs(np.dot(q2, euler_to_quaternion(angles))), 1, atol=1e-12)


def test_uniform_sampler_has_no_preferred_sample_direction():
    q = RandomODF().sample(25_000, np.random.default_rng(123))
    directions = np.einsum("nij,j->ni", quaternion_to_rotation_matrix(q), [0, 0, 1])
    # Isotropy: vector mean is zero and each coordinate's second moment is 1/3.
    assert np.max(np.abs(directions.mean(axis=0))) < .015
    assert np.max(np.abs((directions**2).mean(axis=0) - 1/3)) < .015


def test_random_texture_converges_with_grain_count():
    """Scalable numerical convergence check (100,000 is suitable for a manual run)."""
    errors = []
    for n in (100, 1_000, 10_000):
        q = RandomODF().sample(n, np.random.default_rng(10 + n))
        directions = np.einsum("nij,j->ni", quaternion_to_rotation_matrix(q), [0, 0, 1])
        errors.append(np.linalg.norm(directions.mean(axis=0)))
    assert errors[-1] < errors[0]


def test_component_fiber_mixture_and_empirical_odf_sampling():
    rng = np.random.default_rng(4)
    component = OrientationComponent([1, 0, 0, 0], spread_deg=4)
    assert np.mean(np.abs(SharpOrientationODF().sample(100, rng)[:, 0])) > .999
    q = component.sample(1000, rng)
    assert np.mean(np.abs(q[:, 0])) > .99
    fiber = FiberODF([1, 0, 0], [0, 0, 1], spread_deg=4)
    q_fiber = fiber.sample(1000, rng)
    aligned = np.einsum("nij,j->ni", quaternion_to_rotation_matrix(q_fiber), [1, 0, 0])[:, 2]
    assert aligned.mean() > .99
    mixture = MixtureODF([(0.7, component), (0.3, OrientationComponent([0, 1, 0, 0], 4))])
    q_mix = mixture.sample(5000, rng)
    # The two narrow centers are distinguishable by nearest SO(3) geodesic.
    d_identity = np.arccos(np.clip(np.abs(q_mix @ np.array([1, 0, 0, 0])), 0, 1))
    d_x180 = np.arccos(np.clip(np.abs(q_mix @ np.array([0, 1, 0, 0])), 0, 1))
    assert .65 < np.mean(d_identity < d_x180) < .75
    empirical = estimate_odf(q_mix, spread_deg=5)
    assert empirical.evaluate(np.array([[1, 0, 0, 0]])).item() > empirical.evaluate(np.array([[0, 0, 1, 0]])).item()


def test_symmetry_and_grain_rotation_preserve_reciprocal_lengths():
    assert get_symmetry_operations("cubic").shape == (24, 4)
    assert get_symmetry_operations("hcp").shape == (12, 4)
    grain = GrainCubic(10, 1, 0, 0, 1, 3.6, Experiment(.1, "test"))
    baseline = np.linalg.norm(grain.reciprocal_lattice_vectors, axis=1)
    grain.set_orientation(euler_to_quaternion([.2, .5, 1.1]))
    assert np.allclose(np.linalg.norm(grain.reciprocal_lattice_vectors, axis=1), baseline)


def test_partial_fiber_serialization_and_offset_sampling():
    odf = PartialFiberODF([1, 1, 1], [0.2, 0, 0.98], spread_deg=6,
                          spin_center_deg=40, spin_width_deg=180)
    restored = odf_from_dict(odf.to_dict())
    assert restored.to_dict() == odf.to_dict()
    q = odf.sample(2000, np.random.default_rng(5))
    directions = np.einsum("nij,j->ni", quaternion_to_rotation_matrix(q), [1, 1, 1])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    assert np.mean(directions @ np.array([0.2, 0, 0.98])) > 0.98


def test_continuous_odf_normalization_symmetry_and_sobol_determinism():
    exponent = float(dvp_exponent_from_fwhm(8.0))
    normalizer = float(np.exp(dvp_log_normalization(exponent)))
    integral = quad(lambda theta: normalizer * np.cos(theta / 2) ** (2 * exponent)
                    * (2 / np.pi) * np.sin(theta / 2) ** 2, 0, np.pi,
                    epsabs=1e-10)[0]
    assert np.isclose(integral, 1.0, atol=1e-9)
    space = SobolODFSpace("cubic", max_components=5)
    points_a = sobol_unit_cube(4, space.dimension, seed=99)
    points_b = sobol_unit_cube(4, space.dimension, seed=99)
    assert np.allclose(points_a, points_b)
    odf = space.from_unit_cube(points_a[0])
    q = uniform_quaternions(10, np.random.default_rng(11))
    equivalent = np.array([compose_rotations(item, get_symmetry_operations("cubic")) for item in q])
    assert np.allclose(odf.evaluate(equivalent), odf.evaluate(q)[:, None], atol=2e-10)
    coefficients = harmonic_coefficients(odf, 3)
    assert np.allclose(coefficients[0], [[1.0]], atol=1e-12)
    assert get_symmetry_operations("monoclinic").shape == (2, 4)


def test_grain_shape_and_size_control_peak_width_and_intensity():
    grain = GrainCubic(1000, 0, 0.001, 0, 2.0, 3.6, Experiment(.1, "test"))
    grain.grain_size, grain.grain_strain = 1000., .001
    grain.volume = grain.reference_volume
    long = coherent_length(grain, [0, 0, 1])
    transverse = coherent_length(grain, [1, 0, 0])
    assert long > transverse  # [001] is the declared ellipsoid long axis.
    along_long = peak_properties(grain, [0, 0, 1], .4, .1, 1000)
    along_transverse = peak_properties(grain, [1, 0, 0], .4, .1, 1000)
    assert along_long.fwhm_2theta_rad < along_transverse.fwhm_2theta_rad
    grain.grain_size = 2000.
    larger = peak_properties(grain, [0, 0, 1], .4, .1, 1000)
    assert np.isclose(larger.integrated_intensity, 8 * along_long.integrated_intensity)


def test_configuration_drives_texture_and_grain_realizations():
    config = validate_config({
        "experiment": {"num_grains": 3, "seed": 17},
        "grain": {"size_mean": 1000, "size_std": 0, "aspect_ratio": 2},
        "texture": {"type": "sharp", "spread_deg": 2},
        "detector": {"width_px": 64, "height_px": 64},
    })
    result = run_experiment(config, save=False)
    assert len(result["grains"]) == 3
    assert all(np.isclose(grain["size"], 1000) for grain in result["grains"])
    assert np.all(np.abs(result["orientations"][:, 0]) > .99)


class TestOrientationTexture(unittest.TestCase):
    """stdlib test runner wrapper; these also run unchanged under pytest."""
    def test_quaternions(self): test_quaternion_identity_and_ninety_degree_rotation()
    def test_round_trips(self): test_rotation_and_euler_round_trips()
    def test_uniform(self): test_uniform_sampler_has_no_preferred_sample_direction()
    def test_convergence(self): test_random_texture_converges_with_grain_count()
    def test_odfs(self): test_component_fiber_mixture_and_empirical_odf_sampling()
    def test_symmetry_and_grain(self): test_symmetry_and_grain_rotation_preserve_reciprocal_lengths()
    def test_partial_fiber(self): test_partial_fiber_serialization_and_offset_sampling()
    def test_continuous_odf(self): test_continuous_odf_normalization_symmetry_and_sobol_determinism()
    def test_shape_and_intensity(self): test_grain_shape_and_size_control_peak_width_and_intensity()
    def test_configuration(self): test_configuration_drives_texture_and_grain_realizations()
