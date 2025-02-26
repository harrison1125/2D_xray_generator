class EwaldSphere: 
    """
    I'm Alive! I am the EwaldSphere class
    I know my size and location
    I can project activated diffraction conditions out into all directions. I also know what the intensities of those diffracted points should be. 
    """
    def __init__(self, rotated_grain, experiment, tolerance) :
        self.radius = experiment.inv_lambda
        self.center = (-self.radius, 0, 0)
        self.tolerance = tolerance 
        self.distance_target =  (1/experiment.wavelength) 
        self.rotated_grain = rotated_grain
        #distances of reciprocal lattice vector tips of the grain from the center of the Ewald Sphere. consider erasing center[1] and center[2], as they will likely permanently be 0, and this would just raise computational overhead 
        
        self.distances_of_rlv_from_center = np.sqrt(
            (self.rotated_grain.reciprocal_lattice_vectors[:, 0] - self.center[0])**2 +
            (self.rotated_grain.reciprocal_lattice_vectors[:, 1] - self.center[1])**2 +
            (self.rotated_grain.reciprocal_lattice_vectors[:, 2] - self.center[2])**2
        )

    def filter_points (self):
        mask = (
            (self.distances_of_rlv_from_center >= self.distance_target - self.tolerance) &
            (self.distances_of_rlv_from_center  <= self.distance_target + self.tolerance))
        
        return self.rotated_grain.reciprocal_lattice_vectors[mask]