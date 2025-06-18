# crystal.py
import numpy as np
from scipy.spatial.transform import Rotation as R
import math 

class Experiment:
    def __init__(self, wavelength):
        self.wavelength = wavelength

class GrainCubic:
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

    def randomize_properties (self):
        self.randomize_grain_size
        self.randomize_grain_strain
        self.randomize_rotation



import numpy as np
import math

class Grain_general:
    ''' 
    Handles grain size, strain, and reciprocal lattice vectors for any crystal system.
    Lattice parameters: a, b, c and angles: alpha, beta, gamma (in degrees).
    '''
    def __init__(self, size_average, size_variance, strain_average, strain_variance,
                 aspect_ratio, a, b, c, alpha, beta, gamma, experiment):

        self.size_average = size_average 
        self.size_variance = size_variance 
        self.aspect_ratio = aspect_ratio  # still unused
        self.strain_average = strain_average 
        self.strain_variance = strain_variance

        self.a = a
        self.b = b
        self.c = c
        self.alpha = np.radians(alpha)
        self.beta  = np.radians(beta)
        self.gamma = np.radians(gamma)

        self.sphere_range = math.floor(1 / experiment.wavelength)

        # Generate all integer (h, k, l) indices within the sphere range
        self.hkl_indices = [
            (h, k, l)
            for h in range(-self.sphere_range, self.sphere_range + 1)
            for k in range(-self.sphere_range, self.sphere_range + 1)
            for l in range(-self.sphere_range, self.sphere_range + 1)
        ]

        # Generate reciprocal lattice vectors
        self.reciprocal_lattice_vectors = self._generate_reciprocal_vectors()

    def _generate_reciprocal_vectors(self):
        # Construct the real-space lattice vectors
        a1 = np.array([self.a, 0, 0])
        a2 = np.array([
            self.b * np.cos(self.gamma),
            self.b * np.sin(self.gamma),
            0
        ])
        cx = self.c * np.cos(self.beta)
        cy = self.c * (np.cos(self.alpha) - np.cos(self.beta) * np.cos(self.gamma)) / np.sin(self.gamma)
        cz = np.sqrt(self.c**2 - cx**2 - cy**2)
        a3 = np.array([cx, cy, cz])

        # Volume of the real-space unit cell
        volume = np.dot(a1, np.cross(a2, a3))

        # Reciprocal lattice vectors
        b1 = np.cross(a2, a3) / volume
        b2 = np.cross(a3, a1) / volume
        b3 = np.cross(a1, a2) / volume

        # Generate all reciprocal lattice points using hkl indices
        reciprocal_vectors = np.array([
            h * b1 + k * b2 + l * b3
            for h, k, l in self.hkl_indices
        ])

        return reciprocal_vectors

    def randomize_grain_size(self):
        return np.random.normal(self.size_average, np.sqrt(self.size_variance))
    
    def randomize_grain_strain(self):
        return np.random.normal(self.strain_average, np.sqrt(self.strain_variance))

    def randomize_rotation(self):
        theta = np.radians(np.random.uniform(0, 360))  # z-axis
        phi   = np.radians(np.random.uniform(0, 360))  # x-axis

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

        R = Rx @ Rz  
        self.reciprocal_lattice_vectors = self.reciprocal_lattice_vectors @ R.T

if __name__ == "__main__":
    wavelength = 1.54  # in Angstroms
    lattice_param = 3.615  # FCC copper in Angstroms

    experiment = Experiment(wavelength=wavelength)

    grain_cubic = GrainCubic(
        size_average=100, size_variance=25,
        strain_average=0.001, strain_variance=0.0001,
        aspect_ratio=1,
        lattice_parameter=lattice_param,
        experiment=experiment
    )

    grain_general = GrainGeneral(
        size_average=100, size_variance=25,
        strain_average=0.001, strain_variance=0.0001,
        aspect_ratio=1,
        a=lattice_param, b=lattice_param, c=lattice_param,
        alpha=90, beta=90, gamma=90,
        experiment=experiment
    )

    print("Sample reciprocal vectors (Cubic):")
    for vec in grain_cubic.reciprocal_lattice_vectors[:5]:
        print(vec)

    print("\nSample reciprocal vectors (General):")
    for vec in grain_general.reciprocal_lattice_vectors[:5]:
        print(vec)