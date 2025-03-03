# point_on_detector.py
import numpy as np
import math

# Utility functions moved here to avoid duplications
def atomic_scattering_factor(s, a, b, c):
    """
    Calculate the atomic scattering factor f(s) using provided coefficients.
    """
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Coefficients 'a' and 'b' must each have exactly 4 elements.")
    return sum(a[i] * np.exp(-b[i] * s**2) for i in range(4)) + c

'''
def fwhm_to_sigma(fwhm):
    """
    Convert FWHM to sigma (standard deviation) for a 1D Gaussian.
    """
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def bivariate_gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """
    Evaluate a 2D Gaussian at (x, y) coordinates.
    """
    return amplitude * np.exp(-(((x - x0)**2)/(2 * sigma_x**2) + ((y - y0)**2)/(2 * sigma_y**2)))
'''

class PointOnDetector:
    def __init__(self, grain, ewald_sphere, detector, shape_factor, x_y_coordinate, experiment, exposure_time, a_coeffs, b_coeffs, c_coeff):
        self.grain = grain
        self.wavelength = experiment.wavelength
        self.shape_factor = shape_factor
        self.ewald_sphere = ewald_sphere
        self.detector = detector
        self.exposure_time = exposure_time 
        
        # Use the grain's methods to randomize properties
        self.size = grain.randomize_grain_size()
        self.strain = grain.randomize_grain_strain()
        self.material = "material"  # Placeholder: define based on your sample if needed.
        self.hkl = grain.hkl_indices
        self.projected_points = detector.project_points()

        self.two_theta = np.radians(self.projected_points['two_theta'])
        self.theta = self.two_theta / 2 
        self.lorentz =  {'transmission': 1 / np.sin(2 * self.two_theta), 'reflection': None}
        self.polarization = {"polarized": np.cos(self.two_theta)**2, "unpolarized": (1 + np.cos(2 * self.two_theta)**2) / 2 }
        self.coordinate = x_y_coordinate

        # Calculate broadening effects (placeholders; update as needed)
        self.B_size = (self.shape_factor * self.wavelength) / (self.size * np.cos(self.theta))
        self.B_strain = self.strain * np.tan(self.theta)
        self.B_total = np.sqrt(self.B_size**2 + self.B_strain**2)

        self.s = np.sin(self.theta) / self.wavelength 
        self.f_atomic_scatter = atomic_scattering_factor(self.s, a_coeffs, b_coeffs, c_coeff)

        # Structure factor calculation:
        # Edit: Here structure_factor_func (e.g., from your StructureFactors module) would be applied.
        # For now, we set it as a placeholder.
        self.structure_factor = 1  
        self.structure_factor_conjugate = np.conjugate(self.structure_factor)

        self.total_intensity = (self.structure_factor * self.structure_factor_conjugate *
                                self.lorentz['transmission'] * self.polarization['unpolarized'] *
                                self.exposure_time)
        self.fwhm = 1  # Placeholder value
        self.peak = 1
