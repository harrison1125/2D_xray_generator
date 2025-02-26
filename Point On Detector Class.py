'''

    I'm alive: I'm a point on detector class.
    I know where I am on the detector, what angle I am from the center (the 2theta angle), as well as how bright I am, and how wide I should be, based on inputs from both the grain, the material, and the detector.
    I know how to convolve this value across a single (or in rare cases, multiple) delta functions.

    I know which pixel I'm closest to? - rasterize the image. 

    create a grid that is at least 3x more resolved than the detector itself. Then use a blurring function with a 3x3 pixellation function as the convolution moves acros the detector? This pixellation - how important is it anyhow, especially when synchrotron detectors tend to be so large (megapixel) ?

    I take the detector as an input. At that iteration, the detector is an area where there exists a single delta function. All other points are intensities. Set the Delta function intensity to something very specific and large, something that existing intensities would never be at? Or set the delta function as a non-integer, such as 0.5. 

'''

class PointOnDetector :

    def atomic_scattering_factor (self, s, a, b, c):
        """
        Calculate the atomic scattering factor f(s) using atomic scattering factor coefficients.
        Parameters:
        -----------
        s : float
            Scattering vector magnitude?
        a : list of float
            List of amplitude coefficients [a1, a2, a3, a4].
        b : list of float
            List of exponential damping coefficients [b1, b2, b3, b4].
        c : float
            Constant term for the scattering factor at high s.

        Returns:
        --------
        float
            Atomic scattering factor f(s).
        """
        if len(a) != 4 or len(b) != 4:
            raise ValueError("Coefficients 'a' and 'b' must each have exactly 4 elements.")

        f_s = sum(a[i] * np.exp(-b[i] * s**2) for i in range(4)) + c
        return f_s
        
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
        )) #I wonder if it would be necessary or helpful to apply a correction factor or intensity constant here
        
    def __init__ (self, grain, ewald_sphere, detector, shape_factor, x_y_coordinate, experiment, exposure_time):
        self.grain = grain
        self.wavelength = experiment.wavelength
        self.shape_factor = shape_factor
        self.ewald_sphere = ewald_sphere
        self.detector = detector
        self.wavelength = experiment.wavelength
        self.exposure_time = exposure_time 
        self.size = grain.randomize_grain_size()
        self.strain = grain.randomize_grain_strain()
        self.material = "material"#define grain material for use in the atomic scattering factor 
        self.hkl = grain.hkl_indices
        self.projected_points = detector.project_points()

        self.two_theta = np.radians((self.projected_points['two_theta'])) # check if this is radians or degrees 
        self.theta = self.two_theta / 2 
        self.lorentz =  {'transmission': 1 / np.sin(2 * self.two_theta), 'reflection': None}
        self.polarization = {"polarized": np.cos(self.two_theta)**2, "unpolarized" : (1 + np.cos(2 * self.two_theta)**2) / 2 }
        self.coordinate = x_y_coordinate

        '''check all equations here - there was some buffoonery switching between theta and two_theta, just need to make sure that's all correct '''

        self.B_size = (self.shape_factor * self.wavelength) / (self.size * np.cos(self.theta))
        # Broadening due to strain
        self.B_strain = self.strain * np.tan(self.theta)
        # Total broadening using quadratic sum
        self.B_total = np.sqrt(self.B_size**2 + self.B_strain**2)

        self.s = (np.sin(self.theta))/self.wavelength 
        self.f_atomic_scatter = self.atomic_scattering_factor (self.s, a_cu, b_cu, c_cu)
        self.structure_factor = structure_factor_func(self.grain.hkl_indices[0],self.grain.hkl_indices[1],self.grain.hkl_indices[2], self.f_atomic_scatter)
        self.structure_factor_conjugate = self.structure_factor.conjugate()

        self.total_intensity = self.structure_factor * self.structure_factor_conjugate * self.lorentz['transmission'] * self.polarization['unpolarized'] * self.exposure_time
        self.fwhm = self.B_total**2


        self.fwhm = 1 #Scherrer equation, strain, instrumental?
        self.intensity = self.lorentz * self.polarization * self.structure_factor * self.structure_factor.conj #figure this out later. 
        self.peak = 1