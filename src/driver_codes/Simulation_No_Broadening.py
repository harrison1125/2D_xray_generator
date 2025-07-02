# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Import the modularized classes
from classes_and_functions.experiment import Experiment
from classes_and_functions.crystal import Grain
from classes_and_functions.ewald import EwaldSphere
from classes_and_functions.detector_module import Detector  # EDIT: This is the updated Detector
from classes_and_functions.gauss_param import fwhm_to_sigma, bivariate_gaussian
# 
import StructureFactors  # Your external module for structure factors

# User input for crystal structure.
CrystalStructure = input('What is the crystal structure? ')
structure_factor_map = {
    'SC': StructureFactors.structure_factor_sc, 
    'FCC': StructureFactors.structure_factor_fcc,
    'BCC': StructureFactors.structure_factor_bcc,
    'HCP': StructureFactors.structure_factor_hcp,
}

if CrystalStructure in structure_factor_map:
    structure_factor_func = structure_factor_map[CrystalStructuref]
else:
    structure_factor_func = None  # EDIT: for handling potential errors

# Baseline constants for copper (for atomic scattering factor calculation).
a_cu = [3.5493, 2.6412, 1.5170, 1.0243]
b_cu = [10.2825, 4.2944, 0.2615, 26.1476]
c_cu = 0.2776

# --- Gaussian heatmap parameters ---
FWHM = 10.0          # Full Width at Half Maximum (assumed isotropic)
max_intensity = 3.0  # Peak amplitude
x_center = 65.0      # x-location of the Gaussian center
y_center = 65.0      # y-location of the Gaussian center

# Map extents.
x_min, x_max = 0.0, 100.0
y_min, y_max = 0.0, 100.0

# Figure display settings.
fig_width = 2.4    # inches
fig_height = 2     # inches
dpi = 400        # dots per inch
points_per_unit = 10

# Convert FWHM to sigma.
sigma = fwhm_to_sigma(FWHM)

# Create a meshgrid for the heatmap.
nx = int((x_max - x_min) * points_per_unit)
ny = int((y_max - y_min) * points_per_unit)
x_vals = np.linspace(x_min, x_max, nx)
y_vals = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x_vals, y_vals)
Z = bivariate_gaussian(X, Y, max_intensity, x_center, y_center, sigma, sigma)

# --- Main simulation ---
num_grains = int(input('How many grains? '))
exp = Experiment(wavelength=1.514, sample="Arbitrary")
grain = Grain(size_average=33500, size_variance=33062500, strain_average=0, strain_variance=0, aspect_ratio=1.5, lattice_parameter=0.361, experiment=exp)
coords_on_detector = []

for _ in range(num_grains):
    grain.randomize_rotation()
    # Randomizing size and strain here; the values are generated but not stored externally.
    grain.randomize_grain_size()
    grain.randomize_grain_strain()
    
    ewald = EwaldSphere(grain, exp, tolerance=0.01)
    filtered_points = ewald.filter_points()
    
    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300)
    projected_points = detector.project_points()
    coords_on_detector.extend(projected_points["coordinate"])

coords_on_detector = np.array(coords_on_detector)

# Plot the projected diffraction points.
# EDIT: Updated to use size instead of len, so the "None" case is handled
if coords_on_detector.size > 0:
    x_coords = coords_on_detector[:, 1]
    y_coords = coords_on_detector[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='orange', s=3, label="Projected Points")
    plt.xlim(-detector.detector_width / 2, detector.detector_width / 2)
    plt.ylim(-detector.detector_height / 2, detector.detector_height / 2)
    plt.xlabel("Detector Width (mm)")
    plt.ylabel("Detector Height (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No projected points to display.")

# Plot the 2D Gaussian heatmap.
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap='viridis')
plt.title("2D Gaussian: Grain Size, Two Theta, etc.")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
