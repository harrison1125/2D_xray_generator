# grain.py
import math
import numpy as np

class Grain:
    ''' 
    grain size, lattice parameter, and wavelength need to have the same units, 
    as they all convert into the reciprocal lattice. Any mistakes here will affect 
    the 2theta values. Strain is unitless. 
    '''    
    def __init__(self, size_average, size_variance, strain_average, strain_variance, aspect_ratio, lattice_parameter, experiment):
        self.size_average = size_average 
        self.size_variance = size_variance 
        self.aspect_ratio = aspect_ratio  # Unused currently.
        self.sphere_range = math.floor(1 / experiment.wavelength)
        self.lattice_parameter = lattice_parameter 
        self.strain_average = strain_average 
        self.strain_variance = strain_variance

        # HKL indices for bookkeeping (used later for structure factor calculations)
        self.hkl_indices = [
            (h, k, l)
            for h in range(-self.sphere_range, self.sphere_range + 1)
            for k in range(-self.sphere_range, self.sphere_range + 1)
            for l in range(-self.sphere_range, self.sphere_range + 1)
            # if h**2 + k**2 + l**2 <= self.sphere_range**2  
            # Limiting sphere (Ewald Sphere constraint)

        ]

        # Calculate the reciprocal lattice vectors based on lattice parameter.
        self.reciprocal_lattice_vectors = np.array([
            (
                h / self.lattice_parameter, 
                k / self.lattice_parameter, 
                l / self.lattice_parameter
            )
            for h, k, l in self.hkl_indices
        ])

    '''
    I defined grain_size as a method instead of an attribute so that in later 
    polycrstalline calculations, I could simply re-rotate a reciprocal lattice 
    and change the size repeatedly in order to calculate projections. 
    
    Currently not being used.
    '''

    def randomize_grain_size(self):
        # Edit: Changed "grain.size_average" to "self.size_average"
        # should always refer to the instance's attribute, allowing for multiple grains (reusability)
        return np.random.normal(self.size_average, np.sqrt(self.size_variance))
    
    def randomize_grain_strain(self):
        # Edit: Changed "grain.strain_average" to "self.strain_average"
        # same logic as above
        return np.random.normal(self.strain_average, np.sqrt(self.strain_variance))

    def randomize_rotation(self):
        theta = np.radians(np.random.uniform(0, 360))  # Rotation about z-axis
        phi   = np.radians(np.random.uniform(0, 360))  # Rotation about x-axis

        # Rotation matrices
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)]
        ])

        # Edit: Combined rotation applied to the reciprocal lattice vectors
        R = Rx @ Rz  
        self.reciprocal_lattice_vectors = self.reciprocal_lattice_vectors @ R.T
