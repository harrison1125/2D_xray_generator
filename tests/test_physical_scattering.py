import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.classes_and_functions import StructureFactors
from src.classes_and_functions.crystal import GrainCubic
from src.classes_and_functions.directional_targets import (
    build_directional_targets, reciprocal_plane_normal,
)
from src.classes_and_functions.detector_module import Detector
from src.classes_and_functions.experiment import Experiment
from src.classes_and_functions.grain_scattering import (
    convergence_averaged_excitation_weight, ellipsoid_volume, peak_properties,
    projected_ellipsoid_covariance_px,
)
from src.driver_codes.Simulation_Gaussian_Broadening import (
    _area_bin_image, run_experiment,
)
from src.driver_codes.generate_curated_texture_validation import (
    _png_preview_pixels, generate_validation_corpus,
)
from src.driver_codes.generate_continuous_odf_dataset import (
    _build_design, _make_odf_space, generate as generate_continuous_corpus,
    planned_odf_points, planned_runs,
)


class TestPhysicalScattering(unittest.TestCase):
    def test_png_display_transform_is_monotonic_and_nonmutating(self):
        image = np.array([[0.0, 1.0], [9.0, 99.0]])
        original = image.copy()
        pixels, maximum, display_ceiling = _png_preview_pixels(image)
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(maximum, 99.0)
        self.assertGreater(display_ceiling, 0.0)
        self.assertEqual(pixels[0, 0], 0)
        self.assertEqual(pixels[-1, -1], 255)
        self.assertTrue(np.all(np.diff(pixels.ravel().astype(int)) > 0))
        self.assertTrue(np.array_equal(image, original))

    def test_texture_driver_writes_and_backfills_flat_png_previews(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus"
            spec = {
                "dataset": {"output_directory": str(output), "seed": 8},
                "base_config": {
                    "experiment": {"num_grains": 1},
                    "detector": {"width_px": 32, "height_px": 32},
                    "output": {"store_grains": False, "store_peaks": False},
                },
                "families": [{"name": "proof", "cases": [{"type": "random"}]}],
            }
            self.assertEqual(generate_validation_corpus(spec), 1)
            preview = output / "png_preview" / "proof__case_000__r0000.png"
            self.assertTrue(preview.exists())
            with Image.open(preview) as image:
                self.assertEqual(image.size, (32, 32))
                self.assertEqual(image.info["run_id"], "proof__case_000__r0000")
            preview.unlink()
            self.assertEqual(generate_validation_corpus(spec, resume=True), 0)
            self.assertTrue(preview.exists())

    def test_area_binning_conserves_flux_for_fractional_ratio(self):
        image = np.ones((10, 10), dtype=float)
        binned = _area_bin_image(image, 3, 4)
        self.assertEqual(binned.shape, (3, 4))
        self.assertAlmostEqual(binned.sum(), image.sum())
        self.assertTrue(np.allclose(binned, 100.0 / 12.0))

    def test_lognormal_diameters_are_positive_with_requested_moments(self):
        grain = GrainCubic(10.0, 4.0, 0.0, 0.0, 1.0, 0.361,
                           Experiment(0.0514, "test"),
                           size_distribution="lognormal")
        rng = np.random.default_rng(91)
        samples = np.array([grain.randomize_grain_size(rng) for _ in range(20_000)])
        self.assertTrue(np.all(samples > 0))
        self.assertAlmostEqual(samples.mean(), 10.0, delta=0.05)
        self.assertAlmostEqual(samples.std(), 2.0, delta=0.05)

    def test_micrometre_volume_scherrer_and_excitation_weight(self):
        grain = GrainCubic(33_500.0, 0.0, 0.0, 0.0, 1.0, 0.361,
                           Experiment(0.0514, "Cu"), length_to_microns=1e-3)
        reciprocal_fwhm = 0.9 / grain.grain_size
        properties = peak_properties(
            grain, [0, 0, 1], 0.3, 0.0514, 1000.0,
            instrumental_fwhm_px=1.0,
            excitation_error=reciprocal_fwhm / 2.0,
        )
        self.assertAlmostEqual(properties.volume_um3, ellipsoid_volume(33.5))
        self.assertAlmostEqual(properties.excitation_weight, 0.5)
        expected_scherrer = 0.9 * 0.0514 / (33_500.0 * np.cos(0.15))
        self.assertAlmostEqual(properties.size_fwhm_2theta_rad, expected_scherrer)

    def test_uniform_convergence_integral_matches_numerical_quadrature(self):
        error = 0.01
        reciprocal_fwhm = 3.0e-5
        two_theta = 0.3
        wavelength = 0.0514
        full_angle_mrad = 3.5
        weight, half_span = convergence_averaged_excitation_weight(
            error, reciprocal_fwhm, two_theta, wavelength, full_angle_mrad
        )
        half_angle = 0.5 * full_angle_mrad * 1e-3
        slope = abs(np.sin(two_theta)) / wavelength
        angles = np.linspace(-half_angle, half_angle, 1_000_001)
        numerical = np.trapezoid(
            1.0 / (1.0 + (2.0 * (error + slope * angles) / reciprocal_fwhm) ** 2),
            angles,
        ) / (2.0 * half_angle)
        point_weight, zero_span = convergence_averaged_excitation_weight(
            error, reciprocal_fwhm, two_theta, wavelength, 0.0
        )
        expected_point = 1.0 / (1.0 + (2.0 * error / reciprocal_fwhm) ** 2)
        self.assertAlmostEqual(weight, numerical, places=10)
        self.assertAlmostEqual(half_span, slope * half_angle)
        self.assertAlmostEqual(point_weight, expected_point)
        self.assertEqual(zero_span, 0.0)
        self.assertGreater(weight, 100.0 * point_weight)

    def test_peak_properties_records_configured_convergence_span(self):
        grain = GrainCubic(30_000.0, 0.0, 0.0, 0.0, 1.0, 0.361,
                           Experiment(0.0514, "Cu"), length_to_microns=1e-3)
        properties = peak_properties(
            grain, [1, 1, 1], 0.3, 0.0514, 1000.0,
            excitation_error=0.01,
            incident_convergence_full_angle_mrad=3.5,
        )
        self.assertGreater(properties.convergence_reciprocal_half_span, 0.0)
        self.assertGreater(properties.excitation_weight, 0.0)

    def test_projected_sphere_uses_physical_pixel_pitch(self):
        grain = GrainCubic(33_500.0, 0.0, 0.0, 0.0, 1.0, 0.361,
                           Experiment(0.0514, "Cu"))
        covariance = projected_ellipsoid_covariance_px(
            grain, outgoing_direction=[1, 0, 0], pixel_size=75_000.0
        )
        expected_variance = (33_500.0 / 2.0)**2 / (5.0 * 75_000.0**2)
        self.assertTrue(np.allclose(covariance, np.eye(2) * expected_variance))

    def test_detector_projects_outgoing_ray_from_sample_origin(self):
        grain = GrainCubic(33_500.0, 0.0, 0.0, 0.0, 1.0, 0.361,
                           Experiment(0.1, "test"))

        class Candidates:
            center = (-10.0, 0.0, 0.0)

            def __init__(self, active_grain):
                self.grain = active_grain

            @staticmethod
            def filter_points():
                return np.array([[0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]])

        detector = Detector(Candidates(grain), Experiment(0.1, "test"),
                            100, 100, 100, None,
                            instrumental_fwhm_px=1.0,
                            pixel_size_grain_units=75_000.0)
        result = detector.project_points()
        self.assertTrue(np.allclose(result["coordinate"][0], [100.0, 10.0, 20.0]))
        self.assertAlmostEqual(result["image"].sum(),
                               result["peaks"][0]["integrated_intensity"], places=8)

    def test_configured_micrometre_size_is_converted_explicitly(self):
        result = run_experiment({
            "experiment": {"num_grains": 1},
            "grain": {"size_mean": 33.5, "size_std": 0.0, "size_unit": "um"},
            "detector": {"width_px": 32, "height_px": 32},
        }, save=False)
        self.assertAlmostEqual(result["grains"][0]["size"], 33.5)
        self.assertAlmostEqual(result["grains"][0]["size_nm"], 33_500.0)
        self.assertAlmostEqual(result["grains"][0]["volume_um3"],
                               ellipsoid_volume(33.5))

    def test_native_detector_is_binned_only_after_rendering(self):
        result = run_experiment({
            "experiment": {"num_grains": 1},
            "detector": {
                "width_px": 40, "height_px": 30,
                "bin_to_width_px": 8, "bin_to_height_px": 6,
            },
        }, save=False)
        self.assertEqual(result["image"].shape, (6, 8))
        readout = result["metadata"]["detector_readout"]
        self.assertEqual(readout["native_shape_px"], [30, 40])
        self.assertEqual(readout["output_shape_px"], [6, 8])
        self.assertEqual(readout["binning_ratio"], [5.0, 5.0])

    def test_continuous_corpus_reuses_exact_odf_label_across_grain_counts(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "continuous"
            spec = {
                "run": {
                    "output_directory": str(output),
                    "design_seed": 41,
                    "observation_seed": 42,
                    "num_samples_per_symmetry": 2,
                    "grain_counts": [1, 2],
                    "sampling_method": "Sobol_sequence",
                    "kernel_type": "de_la_vallee_poussin",
                    "max_texture_components": 5,
                    "output_format": "parameters_only",
                    "write_tiff": False,
                    "directional_targets": {
                        "enabled": True,
                        "source": "observation_orientations",
                        "grid_size": 8,
                        "coefficient_shape": [2, 2],
                        "sample_directions": [{"name": "ND", "direction": [0, 0, 1]}],
                    },
                },
                "base_config": {
                    "experiment": {"num_grains": 1, "max_hkl_index": 1},
                    "detector": {"width_px": 32, "height_px": 32},
                    "output": {
                        "store_grains": False, "store_peaks": False,
                        "image_format": "npy",
                    },
                },
                "crystal_systems": [{
                    "name": "cubic_fcc", "orientation_symmetry": "cubic",
                    "pole_families": [{"name": "{111}", "hkl": [1, 1, 1]}],
                    "simulation": {"experiment": {
                        "material": "Cu", "crystal_structure": "FCC",
                        "lattice_parameter": 0.361,
                    }},
                }],
            }
            self.assertEqual(planned_odf_points(spec), 2)
            self.assertEqual(planned_runs(spec), 4)
            self.assertEqual(generate_continuous_corpus(spec), 4)

            with np.load(output / "cubic_fcc" / "odf_labels.npz") as labels:
                self.assertEqual(labels["descriptor"].shape, (2, 31))
                first_descriptor = labels["descriptor"][0].copy()
                self.assertAlmostEqual(
                    float(labels["background_weight"][0]
                          + labels["component_weights"][0].sum()), 1.0,
                )
            with np.load(output / "planned_scan_index.npz") as index:
                self.assertEqual(index["odf_descriptor"].shape, (4, 31))
                self.assertTrue(np.array_equal(index["grain_count"], [1, 2, 1, 2]))
                self.assertTrue(np.array_equal(index["odf_descriptor"][0],
                                               index["odf_descriptor"][1]))
                self.assertIn("pf_ipf_path", index)
                self.assertIn("design_pair_id", index)
            label_paths = sorted((output / "cubic_fcc").glob("*g00000*/odf_label.npz"))
            self.assertEqual(len(label_paths), 4)
            with np.load(label_paths[0]) as label:
                self.assertTrue(np.array_equal(label["descriptor"], first_descriptor))
                self.assertEqual(str(label["label_version"]),
                                 "continuous_symmetric_dvp_mixture_v1")
            directional_paths = sorted(
                (output / "cubic_fcc").glob("*g00000*/pf_ipf_targets.npz")
            )
            self.assertEqual(len(directional_paths), 4)
            with np.load(directional_paths[0]) as targets:
                self.assertEqual(targets["directional_coefficients"].shape, (2, 4))
                self.assertTrue(np.allclose(
                    targets["pole_figure_probability_mass"].sum(axis=(1, 2)), 1.0
                ))
                self.assertTrue(np.allclose(
                    targets["inverse_pole_figure_probability_mass"].sum(axis=(1, 2)), 1.0
                ))
            records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual({record["grain_count"] for record in records}, {1, 2})
            self.assertEqual(len({record["seed"] for record in records}), 4)

    def test_directional_targets_use_reciprocal_hcp_geometry_and_symmetry(self):
        cell = {
            "a": 0.295, "b": 0.295, "c": 0.4683,
            "alpha_deg": 90.0, "beta_deg": 90.0, "gamma_deg": 120.0,
        }
        self.assertTrue(np.allclose(reciprocal_plane_normal([0, 0, 2], cell), [0, 0, 1]))
        identity = np.tile([1.0, 0.0, 0.0, 0.0], (6, 1))
        targets = build_directional_targets(
            identity,
            crystal_symmetry="hexagonal",
            pole_families=[
                {"name": "{0002}", "hkl": [0, 0, 2]},
                {"name": "{10-10}", "hkl": [1, 0, 0]},
            ],
            sample_directions=[{"name": "ND", "direction": [0, 0, 1]}],
            grid_size=16,
            coefficient_shape=(4, 4),
            unit_cell=cell,
        )
        self.assertEqual(targets["directional_coefficients"].shape, (3, 16))
        self.assertTrue(np.allclose(
            targets["pole_figure_probability_mass"].sum(axis=(1, 2)), 1.0
        ))
        self.assertTrue(np.allclose(
            targets["inverse_pole_figure_probability_mass"].sum(axis=(1, 2)), 1.0
        ))

    def test_hcp_cell_and_basis_are_used_by_simulator(self):
        cell = {
            "a": 0.295, "b": 0.295, "c": 0.4683,
            "alpha_deg": 90.0, "beta_deg": 90.0, "gamma_deg": 120.0,
        }
        result = run_experiment({
            "experiment": {
                "material": "Ti", "crystal_structure": "HCP",
                "lattice_parameter": 0.295, "unit_cell": cell,
                "num_grains": 1, "max_hkl_index": 1,
            },
            "detector": {"width_px": 16, "height_px": 16},
            "output": {"store_grains": False, "store_peaks": False},
        }, save=False)
        self.assertEqual(result["metadata"]["grain_lattice_model"], "GrainGeneral")
        self.assertAlmostEqual(abs(StructureFactors.structure_factor_hcp(0, 0, 1, 1)), 0.0)
        self.assertAlmostEqual(abs(StructureFactors.structure_factor_hcp(0, 0, 2, 1)), 2.0)

    def test_physical_design_supports_soft_likelihood_and_hard_prior_modes(self):
        system = {"name": "cubic", "orientation_symmetry": "cubic"}
        broad_run = {
            "sampling_method": "physical_sobol_manifold",
            "max_texture_components": 5,
            "physical_texture": {
                "inference_mode": "likelihood",
                "process_family": "rolling",
                "texture_index_policy": "soft_weight",
                "texture_index_bounds": [1.1, 35.0],
            },
        }
        broad_space = _make_odf_space(broad_run, system)
        points, metadata = _build_design(8, broad_space, 17, broad_run, system)
        self.assertEqual(points.shape, (8, 27))
        self.assertEqual(metadata["component_manifold"].shape, (8, 5))
        self.assertTrue(np.all(np.isfinite(metadata["texture_index"])))
        self.assertTrue(np.all(metadata["texture_index_log_weight"] <= 0))

        prior_run = {
            "sampling_method": "physical_sobol_manifold",
            "max_texture_components": 5,
            "physical_texture": {
                "inference_mode": "physical_prior",
                "process_family": "rolling",
                "texture_index_policy": "hard_reject",
                "texture_index_bounds": [1.1, 35.0],
                "hard_rejection_candidate_factor": 8,
            },
        }
        prior_space = _make_odf_space(prior_run, system)
        _, prior_metadata = _build_design(8, prior_space, 18, prior_run, system)
        self.assertTrue(np.all(prior_metadata["texture_index"] >= 1.1))
        self.assertTrue(np.all(prior_metadata["texture_index"] <= 35.0))

    def test_sputter_pvd_catalog_has_requested_component_balance_and_support(self):
        spec = json.loads(Path("configs/continuous_odf_sputter_pvd_broad.json").read_text())
        run = spec["run"]
        for system in spec["crystal_systems"]:
            space = _make_odf_space(run, system)
            points, metadata = _build_design(512, space, 423, run, system)
            self.assertEqual(points.shape, (512, 33))
            self.assertEqual(set(metadata["active_component_count"].tolist()), set(range(1, 7)))

            tags, widths, contains_film_texture, biaxial_indices = [], [], [], []
            for index, (point, origin, mask) in enumerate(zip(
                    points, metadata["component_process_tag"],
                    metadata["component_active_mask"])):
                odf = space.from_unit_cube(point)
                active = np.asarray(mask, dtype=bool)
                active_tags = np.asarray(origin)[active]
                tags.extend(active_tags.tolist())
                widths.extend(odf.component_fwhm_deg[active].tolist())
                contains_film_texture.append(
                    np.any(active_tags != "nanocrystalline_uniform")
                )
                if np.any(active_tags == "sputter_biaxial"):
                    biaxial_indices.append(index)

            proportions = {tag: tags.count(tag) / len(tags) for tag in set(tags)}
            self.assertAlmostEqual(proportions["sputter_fiber"], 0.65, delta=0.03)
            self.assertAlmostEqual(proportions["sputter_biaxial"], 0.20, delta=0.03)
            self.assertAlmostEqual(proportions["nanocrystalline_uniform"], 0.15, delta=0.03)
            self.assertGreater(float(np.mean(contains_film_texture)), 0.80)
            self.assertGreater(float(metadata["texture_index"][biaxial_indices].max()), 10.0)
            self.assertGreaterEqual(min(widths), 5.0)
            self.assertLessEqual(max(widths), 30.0)


if __name__ == "__main__":
    unittest.main()
