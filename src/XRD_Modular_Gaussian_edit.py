# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Import the modularized classes
from experiment_module import Experiment
from sample_module import Sample
from grain_module import Grain
from e_sphere_module import EwaldSphere
from detector_module import Detector  # EDIT: This is the updated Detector
from gauss_param import fwhm_to_sigma, bivariate_gaussian

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

# Initialize the composite image and coordinates
coords_on_detector = []
composite_image = np.zeros((1000, 1000))


first_grain = Grain(
    size_average=33500, 
    size_variance=33062500, 
    strain_average=0, 
    strain_variance=0, 
    aspect_ratio=1.5, 
    lattice_parameter=0.361, 
    experiment=exp
)
first_grain.randomize_rotation()
first_grain.randomize_grain_size()
first_grain.randomize_grain_strain()
initial_ewald = EwaldSphere(first_grain, exp, tolerance=0.03)

detector = Detector(initial_ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300)

for grain_index in range(num_grains):
    grain.randomize_rotation()
    grain.randomize_grain_size()
    grain.randomize_grain_strain()

<<<<<<< Updated upstream
    ewald = EwaldSphere(grain, exp, tolerance=0.03)
    detector.ewald_sphere = ewald  # update the Ewald sphere for the current grain
    projected_points = detector.project_points()  # This updates detector.image

    # Debugging: Print the sum of the grain-specific image
    # print(f"Grain {grain_index + 1}: Grain Image Sum = {np.sum(projected_points['image'])}")
=======
    ewald = EwaldSphere(grain, exp, tolerance=0.01)
    # ADDED: Use the updated Detector that produces Gaussian spots
    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300)
    projected_points = detector.project_points()
>>>>>>> Stashed changes

    # Collect the raw (x, y, z) coordinates for a scatter plot if desired
    coords_on_detector.extend(projected_points["coordinate"])

    # Accumulate the Gaussian intensities from each grain into composite_image
    composite_image += projected_points["image"]

    # Debugging: Print the composite image sum after each grain
    # print(f"Composite Image Sum After Grain {grain_index + 1}: {np.sum(composite_image)}")

# Convert coordinates to a NumPy array after the loop
coords_on_detector = np.array(coords_on_detector)
#plt.imshow(composite_image_normalized, cmap='hot', origin='lower', extent=[-500, 500, -500, 500])

# --- Block the central spot using Option 1: Completely block it ---
# Define the blocker FWHM (in pixels) based on the central spot's size (adjust as needed)
blocker_fwhm = 35.0 
blocker_radius = blocker_fwhm / 2.0

# Determine the center of the composite image
center_row = composite_image.shape[0] // 2
center_col = composite_image.shape[1] // 2

# Create a grid of indices corresponding to each pixel
yy, xx = np.indices(composite_image.shape)

# Compute the distance of each pixel from the center
distance_from_center = np.sqrt((xx - center_col)**2 + (yy - center_row)**2)

# Create a binary mask: True for pixels outside the blocker region, False inside
mask = distance_from_center > blocker_radius

# Option 1: Completely block the center (set intensities to zero)
composite_image_blocked = composite_image.copy()
composite_image_blocked[~mask] = 0

# Plot the final composite detector image
plt.figure(figsize=(8, 8))
plt.imshow(np.log1p(composite_image_blocked), cmap='hot', origin='lower', extent=[-500, 500, -500, 500])
plt.xlabel("Detector Width (mm)")
plt.ylabel("Detector Height (mm)")
plt.title("Final Composite Detector Image with Gaussian Distributed Spots")
plt.colorbar(label='Intensity')
plt.show()

'''
# Plot the example 2D Gaussian heatmap (unchanged)
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap='viridis')
plt.title("2D Gaussian: Grain Size, Two Theta, etc.")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
'''

'''
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
'''

