# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Import the modularized classes
from experiment import Experiment
from sample import Sample
from crystal import Grain
from ewald import EwaldSphere
from experiment import Experiment, Detector  # EDIT: This is the updated Detector
from broadening_related.gauss_param import fwhm_to_sigma, bivariate_gaussian

import StructureFactors  # For structure factor computations

# --------------------------------------------------
# The next lines for structure factor / user input remain unchanged
# --------------------------------------------------

CrystalStructure = input('What is the crystal structure? ')
structure_factor_map = {
    'SC': StructureFactors.structure_factor_sc, 
    'FCC': StructureFactors.structure_factor_fcc,
    'BCC': StructureFactors.structure_factor_bcc,
    'HCP': StructureFactors.structure_factor_hcp,
}

if CrystalStructure in structure_factor_map:
    structure_factor_func = structure_factor_map[CrystalStructure]
else:
    structure_factor_func = None

# Baseline constants for copper, unchanged
a_cu = [3.5493, 2.6412, 1.5170, 1.0243]
b_cu = [10.2825, 4.2944, 0.2615, 26.1476]
c_cu = 0.2776

# --- Example for a single 2D Gaussian heatmap (unchanged) ---
FWHM = 10.0
max_intensity = 3.0
x_center = 65.0
y_center = 65.0

x_min, x_max = 0.0, 100.0
y_min, y_max = 0.0, 100.0

fig_width = 2.4
fig_height = 2
dpi = 400
points_per_unit = 10

sigma = fwhm_to_sigma(FWHM)
nx = int((x_max - x_min) * points_per_unit)
ny = int((y_max - y_min) * points_per_unit)
x_vals = np.linspace(x_min, x_max, nx)
y_vals = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x_vals, y_vals)
Z = bivariate_gaussian(X, Y, max_intensity, x_center, y_center, sigma, sigma)

# --- Main simulation ---
num_grains = int(input('How many grains? '))
exp = Experiment(wavelength=0.154, sample="Arbitrary")

grain = Grain(
    size_average=33500, 
    size_variance=33062500, 
    strain_average=0, 
    strain_variance=0, 
    aspect_ratio=1.5, 
    lattice_parameter=0.361, 
    experiment=exp
)

# ADDED: We will accumulate the projected coordinates AND sum a composite detector image
coords_on_detector = []
# ADDED: Initialize a composite image the same size as the detector
composite_image = np.zeros((1000, 1000))

for _ in range(num_grains):
    grain.randomize_rotation()
    grain.randomize_grain_size()
    grain.randomize_grain_strain()

    ewald = EwaldSphere(grain, exp, tolerance=0.03)
    # ADDED: Use the updated Detector that produces Gaussian spots
    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300)
    projected_points = detector.project_points()

    # Collect the raw (x, y, z) coordinates for a scatter plot if desired
    coords_on_detector.extend(projected_points["coordinate"])
    # ADDED: Accumulate the Gaussian intensities from each grain into composite_image
    composite_image += projected_points["image"]

coords_on_detector = np.array(coords_on_detector)

# Plot the raw projected points (optional)
if coords_on_detector.size > 0:
    x_coords = coords_on_detector[:, 1]
    y_coords = coords_on_detector[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='orange', s=3, label="Projected Points")
    # The detector is 1000 mm wide/high, so set the axes accordingly:
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.xlabel("Detector Width (mm)")
    plt.ylabel("Detector Height (mm)")
    plt.legend()
    plt.grid(True)
    plt.title("Scatter Plot of Projected Diffraction Points")
    plt.show()
else:
    print("No projected points to display.")

filtered_points = ewald.filter_points()
print("Number of reflections in this grain:", len(filtered_points))

# ADDED: Plot the composite detector image with Gaussian-distributed spots
plt.figure(figsize=(8, 8))
# 'origin=lower' to put (0,0) at bottom-left, 'extent' maps pixel coords to physical mm
plt.imshow(composite_image, cmap='hot', origin='lower',
           extent=[-500, 500, -500, 500])
plt.xlabel("Detector Width (mm)")
plt.ylabel("Detector Height (mm)")
plt.title("Detector Image with Gaussian Distributed Spots")
plt.colorbar(label='Intensity')
plt.show()

# Plot the example 2D Gaussian heatmap (unchanged)
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap='viridis')
plt.title("2D Gaussian: Grain Size, Two Theta, etc.")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
