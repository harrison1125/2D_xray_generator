"""Grain lattice objects with quaternion orientations and realized properties."""
import math
import numpy as np
from .orientation import (canonicalize_quaternion, quaternion_to_rotation_matrix,
                          uniform_quaternions)
from .grain_scattering import ellipsoid_volume
from .experiment import Experiment

class GrainCubic:
    ''' 
    grain size, lattice parameter, and wavelength need to have the same units, 
    as they all convert into the reciprocal lattice. Any mistakes here will affect 
    the 2theta values. Strain is unitless. 
    '''    
    def __init__(self, size_average, size_variance, strain_average, strain_variance, aspect_ratio, lattice_parameter, experiment):
        self.size_average = size_average 
        self.size_variance = size_variance 
        self.aspect_ratio = aspect_ratio
        self.sphere_range = math.floor(1 / experiment.wavelength)
        self.lattice_parameter = lattice_parameter 
        self.strain_average = strain_average 
        self.strain_variance = strain_variance
        # Per-realization values are populated by ``realize_properties``.
        self.grain_size = size_average
        self.grain_strain = strain_average
        self.volume = ellipsoid_volume(size_average)
        self.reference_volume = self.volume

        # HKL indices for bookkeeping (used later for structure factor calculations)
        self.hkl_indices = [
            # (h, k, l)
            # for h in range(-self.sphere_range, self.sphere_range + 1)
            # for k in range(-self.sphere_range, self.sphere_range + 1)
            # for l in range(-self.sphere_range, self.sphere_range + 1)
            # if h**2 + k**2 + l**2 <= self.sphere_range**2  
            # Limiting sphere (Ewald Sphere constraint)
            (h, k, l)
             for h in range(-4, 5)
             for k in range(-4, 5)
             for l in range(-4, 5)


        ]

        # Calculate the reciprocal lattice vectors based on lattice parameter.
        self._crystal_reciprocal_lattice_vectors = np.array([
            (
                h / self.lattice_parameter, 
                k / self.lattice_parameter, 
                l / self.lattice_parameter
            )
            for h, k, l in self.hkl_indices
        ])
        self.orientation = np.array([1., 0., 0., 0.])
        self._update_oriented_reciprocal_lattice()

    '''
    I defined grain_size as a method instead of an attribute so that in later 
    polycrstalline calculations, I could simply re-rotate a reciprocal lattice 
    and change the size repeatedly in order to calculate projections. 
    
    Currently not being used.
    '''

    def randomize_grain_size(self, rng=None):
        # Edit: Changed "grain.size_average" to "self.size_average"
        # should always refer to the instance's attribute, allowing for multiple grains (reusability)
        return (np.random.default_rng() if rng is None else rng).normal(self.size_average, np.sqrt(self.size_variance))
    
    def randomize_grain_strain(self, rng=None):
        # Edit: Changed "grain.strain_average" to "self.strain_average"
        # same logic as above
        return (np.random.default_rng() if rng is None else rng).normal(self.strain_average, np.sqrt(self.strain_variance))

    def _update_oriented_reciprocal_lattice(self):
        """Derive sample-frame reciprocal vectors from immutable crystal vectors."""
        matrix = quaternion_to_rotation_matrix(self.orientation)
        self.reciprocal_lattice_vectors = self._crystal_reciprocal_lattice_vectors @ matrix.T

    def set_orientation(self, quaternion):
        """Set active crystal-to-sample ``[w,x,y,z]`` orientation."""
        self.orientation = canonicalize_quaternion(np.asarray(quaternion, dtype=float))
        self._update_oriented_reciprocal_lattice()

    def randomize_rotation(self, rng=None):
        """Backward-compatible random orientation using Haar-uniform SO(3)."""
        self.set_orientation(uniform_quaternions(1, rng)[0])
        return self.orientation

    def realize_properties(self, rng=None):
        """Sample and retain this grain's size, strain, and volume."""
        rng = np.random.default_rng() if rng is None else rng
        # A normal distribution can produce a nonphysical negative tail.
        self.grain_size = max(np.finfo(float).eps, self.randomize_grain_size(rng))
        self.grain_strain = self.randomize_grain_strain(rng)
        self.volume = ellipsoid_volume(self.grain_size)
        return self.grain_size, self.grain_strain

    def randomize_properties(self, rng=None):
        """Backward-compatible convenience method that retains all sampled values."""
        self.randomize_rotation(rng)
        return self.realize_properties(rng)







class GrainGeneral:
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
        self.grain_size = size_average
        self.grain_strain = strain_average
        self.volume = ellipsoid_volume(size_average)
        self.reference_volume = self.volume

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
        self._crystal_reciprocal_lattice_vectors = self._generate_reciprocal_vectors()
        self.orientation = np.array([1., 0., 0., 0.])
        self._update_oriented_reciprocal_lattice()

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

    def randomize_grain_size(self, rng=None):
        return (np.random.default_rng() if rng is None else rng).normal(self.size_average, np.sqrt(self.size_variance))
    
    def randomize_grain_strain(self, rng=None):
        return (np.random.default_rng() if rng is None else rng).normal(self.strain_average, np.sqrt(self.strain_variance))

    def _update_oriented_reciprocal_lattice(self):
        matrix = quaternion_to_rotation_matrix(self.orientation)
        self.reciprocal_lattice_vectors = self._crystal_reciprocal_lattice_vectors @ matrix.T

    def set_orientation(self, quaternion):
        self.orientation = canonicalize_quaternion(np.asarray(quaternion, dtype=float))
        self._update_oriented_reciprocal_lattice()

    def randomize_rotation(self, rng=None):
        self.set_orientation(uniform_quaternions(1, rng)[0])
        return self.orientation

    def realize_properties(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.grain_size = max(np.finfo(float).eps, self.randomize_grain_size(rng))
        self.grain_strain = self.randomize_grain_strain(rng)
        self.volume = ellipsoid_volume(self.grain_size)
        return self.grain_size, self.grain_strain

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
