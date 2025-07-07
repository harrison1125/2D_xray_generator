# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys 
import copy

# Import the modularized classes
from src.classes_and_functions.experiment import Experiment
from src.classes_and_functions.crystal import GrainCubic
from src.classes_and_functions.ewald import EwaldSphere
from src.classes_and_functions.detector_module import Detector  # EDIT: This is the updated Detector
from src.classes_and_functions.gauss_param import fwhm_to_sigma, bivariate_gaussian
from src.classes_and_functions import StructureFactors

# import StructureFactors  # For structure factor computations
import json
import datetime 
import random
import string
import os
import imageio.v2 as imageio
from PIL import TiffImagePlugin, Image
import tifffile

def run_from_cli():
    material_type = input('What is the material? ')
    CrystalStructure = input('What is the crystal structure? ')
    num_grains = int(input('How many grains? '))
    lattice_parameter = float(input('What is the lattice parameter? '))
    
    diffraction_pattern(material_type, CrystalStructure, num_grains, lattice_parameter)

def diffraction_pattern(material_type, CrystalStructure, num_grains, lattice_parameter):

    destination = 'output/'
    inputs = {}
    inputs['material_type'] = material_type
    inputs['num_grains'] = num_grains
    wavelength = 0.0514
    inputs['wavelength'] = wavelength
    
    # 4-character random hex-like ID
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    inputs['id'] = random_id

    today = datetime.datetime.today()
    # Format as MMDDYYYY
    formatted_date = today.strftime("%Y%m%d")
    inputs['formatted_date'] = formatted_date

    # CrystalStructure = input('What is the crystal structure? ')
    structure_factor_map = {
        'SC': StructureFactors.structure_factor_sc, 
        'FCC': StructureFactors.structure_factor_fcc,
        'BCC': StructureFactors.structure_factor_bcc,
        'HCP': StructureFactors.structure_factor_hcp,
    }
    
    if CrystalStructure in structure_factor_map:
        structure_factor_func = structure_factor_map[CrystalStructure]
    else:
        raise ValueError(f'Crystal Structure value of {CrystalStructure} is not elligble.')

    inputs['crystal_structure'] = CrystalStructure

    # num_grains = int(input('How many grains? '))
    exp = Experiment(wavelength=wavelength, sample=material_type)

    size_average = 33500
    size_variance = 33062500
    strain_average = 0
    strain_variance = 0
    aspect_ratio = 1.5

    inputs['lattice_parameter'] = lattice_parameter

    grain = GrainCubic(
        size_average=size_average, 
        size_variance=size_variance, 
        strain_average=strain_average, 
        strain_variance=strain_variance, 
        aspect_ratio=aspect_ratio, 
        lattice_parameter = lattice_parameter,
        experiment=exp
    )

    # Initialize the composite image and coordinates
    coords_on_detector = []
    detector_height =  1000
    detector_width = 1000
    composite_image = np.zeros((detector_height, detector_width))

    inputs['detector_height'] = detector_height
    inputs['detector_width'] = detector_width

    tolerance = 0.1
    detector_distance_in_mm = 86 # mm
    detector_distance_in_px = detector_distance_in_mm / 0.075 # pixels

    inputs['tolerance'] = tolerance
    inputs['detector_distance_in_px'] = detector_distance_in_px

    for grain_index in range(num_grains):
        grain.randomize_rotation()
        #grain.randomize_grain_size()
        grain.randomize_grain_strain()

        ewald = EwaldSphere(grain, exp, tolerance=tolerance)

        # ADDED: Use the updated Detector that produces Gaussian spots
        detector = Detector(ewald, exp, detector_width=detector_width, detector_height=detector_height, detector_distance=detector_distance_in_px, structure_factor_func = structure_factor_func)
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
    # plt.figure(figsize=(8, 8))
    # plt.imshow(np.log1p(composite_image_blocked), cmap='hot', origin='lower', extent=[-500, 500, -500, 500])
    # plt.xlabel("Detector Width (mm)")
    # plt.ylabel("Detector Height (mm)")
    # plt.title("Final Composite Detector Image with Gaussian Distributed Spots")
    # plt.colorbar(label='Intensity')
    # plt.show()
    
    naming = f'{material_type}_{CrystalStructure}_{num_grains}_{formatted_date}_{random_id}'

    # Convert to JSON string
    metadata_str = json.dumps(inputs, indent=4)

    # Save to file
    inputs_path = f'{naming}_inputs.json'
    inputs_path = os.path.join(destination, inputs_path)
    with open(inputs_path, 'w') as f:
        f.write(metadata_str)
    
    # Convert to float32 image
    float_image = composite_image_blocked.astype(np.float32)

    # Save the image + metadata using tifffile
    tiff_path = f"{naming}.tiff"
    tiff_path = os.path.join(destination, tiff_path)
    tifffile.imwrite(tiff_path,
            float_image,
            dtype=np.float32,
            metadata={'ImageDescription': metadata_str}
            )
    
    with tifffile.TiffFile(tiff_path) as tif:
        image_data = tif.asarray()
        meta = tif.pages[0].tags['ImageDescription'].value
        metadata_dict = json.loads(meta)
        print("Recovered metadata:", metadata_dict)

if __name__ == "__main__":
    run_from_cli()
