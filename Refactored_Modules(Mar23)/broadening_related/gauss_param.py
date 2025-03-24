import numpy as np

def fwhm_to_sigma(fwhm):
    """
    Convert FWHM to standard deviation sigma for a 1D Gaussian.
    FWHM = 2 * sqrt(2 * ln(2)) * sigma
    """
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def bivariate_gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """
    Returns the 2D Gaussian value at each point in the meshgrid (x, y).
    No correlation term, so it factors into (x part) * (y part).
    """
    return amplitude * np.exp(-(
        ((x - x0)**2)/(2 * sigma_x**2) + ((y - y0)**2)/(2 * sigma_y**2)
    ))  