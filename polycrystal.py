import numpy as np
from crystal import Grain  # Import Grain class


class Polycrystal:
    def __init__(self, num_grains, lattice_parameters):
        """
        Represents a polycrystalline material with multiple grains.

        :param num_grains: Number of grains in the polycrystal
        :param lattice_parameters: Lattice parameters shared by all grains
        """
        self.num_grains = num_grains
        self.lattice_parameters = lattice_parameters
        self.grains = [Grain(lattice_parameters) for _ in range(num_grains)]

    def get_combined_diffraction_pattern(self, wavelength):
        """
        Simulates the combined diffraction pattern of all grains.

        :param wavelength: X-ray wavelength (in same units as lattice parameters)
        :return: List of diffraction spots from all grains
        """
        diffraction_pattern = []

        for grain in self.grains:
            diffraction_spots = grain.get_diffraction_spots(wavelength)
            diffraction_pattern.extend(diffraction_spots)

        return np.array(diffraction_pattern)
