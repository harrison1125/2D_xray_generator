# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys 
import copy

# Import the modularized classes
from classes_and_functions.experiment import Experiment
from classes_and_functions.crystal import GrainCubic
from classes_and_functions.ewald import EwaldSphere
from classes_and_functions.detector_module import Detector  # EDIT: This is the updated Detector
from classes_and_functions.gauss_param import fwhm_to_sigma, bivariate_gaussian

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

# --- Main simulation ---
num_grains = int(input('How many grains? '))
exp = Experiment(wavelength=0.0514, sample="Arbitrary")

# grain = GrainGeneral(
#     size_average=33500, 
#     size_variance=33062500, 
#     strain_average=0, 
#     strain_variance=0, 
#     aspect_ratio=1.5, 
#     a=0.332, 
#     b=0.332, 
#     c=0.332, 
#     alpha=90, 
#     beta=90, 
#     gamma=90, 
#     experiment=exp
# )

grain = GrainCubic(
    size_average=33500, 
    size_variance=33062500, 
    strain_average=0, 
    strain_variance=0, 
    aspect_ratio=1.5, 
    lattice_parameter = 0.3615,
    experiment=exp
)

# Initialize the composite image and coordinates
coords_on_detector = []
composite_image = np.zeros((1000,1000))


# first_grain = GrainGeneral(
#     size_average=33500, 
#     size_variance=33062500, 
#     strain_average=0, 
#     strain_variance=0, 
#     aspect_ratio=1.5, 
#     a=0.308, 
#     b=0.308, 
#     c=0.308, 
#     alpha=90, 
#     beta=90, 
#     gamma=90, 
#     experiment=exp
# )
first_grain = GrainCubic(
    size_average=33500, 
    size_variance=33062500, 
    strain_average=0, 
    strain_variance=0, 
    aspect_ratio=1.5, 
    lattice_parameter = 0.3615,
    experiment=exp
)

rotated_grain = first_grain.randomize_rotation()
first_grain.randomize_grain_size()
first_grain.randomize_grain_strain()
initial_ewald = EwaldSphere(rotated_grain, exp, tolerance=0.03)



detector = Detector(initial_ewald, exp, detector_width=1000, detector_height=1000, detector_distance=1147, structure_factor_func = structure_factor_func)
#currently using 1147 as the detector distance. Seeing as detector width/height are based in pixes, the chosen distance right now represents our detector distance had we calculated it in pixels (86 mm distance, 75 um pixel size: sample-to-detector distance = 1147 pixels)

for grain_index in range(num_grains):
    rotated_grain= grain.randomize_rotation()
    grain.randomize_grain_size()
    grain.randomize_grain_strain()

    ewald = EwaldSphere(rotated_grain, exp, tolerance=0.03)

    # ADDED: Use the updated Detector that produces Gaussian spots
    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=1147, structure_factor_func = structure_factor_func)
    projected_points = detector.project_points()

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
blocker_fwhm = 5.0 
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

if __name__ == "__main__":
    # do something 
    print('1')