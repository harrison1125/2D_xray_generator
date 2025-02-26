'''Grain Class

Houses all the information related to the specific grain that is currently getting examined. Designed so that it can easily be iterated over multiple grains, which would allow us to have different numbers of grains of different types (secondary phases, materials, structures) where necessary. 

I'm Alive statement: I am the Grain class. I know my properties based on traits such as space group, Wyckoff positions, and lattice parameters. I can rotate and adjust my strain and size (specifically for polycrystalline simulations).'''

class Grain: 

    ''' grain size, lattice parameter, and wavelength need to have the same units, as they all convert into the reciprocal lattice. Any mistakes here will affect the 2theta values. Strain is unitless. '''    
    def __init__(self, size_average, size_variance, strain_average, strain_variance, aspect_ratio, lattice_parameter, experiment):
        self.size_average = size_average 
        self.size_variance = size_variance 
        self.aspect_ratio = aspect_ratio # Unused. 
        self.sphere_range = math.floor(1/ experiment.wavelength)
        self.lattice_parameter = lattice_parameter 
        self.strain_average = strain_average 
        self.strain_variance = strain_variance

        '''HKL indices are necessary as book-keeping for later calculations of structure factor''' 
        self.hkl_indices = [
            (h, k, l)
            for h in range(-self.sphere_range, self.sphere_range + 1)
            for k in range(-self.sphere_range, self.sphere_range + 1)
            for l in range(-self.sphere_range, self.sphere_range + 1)
        #    if h**2 + k**2 + l**2 <= self.sphere_range**2  # Limiting sphere (Ewald Sphere constraint)
        ]

        '''Actual reciprocal lattice coordinates in reciprocal space.'''
        self.reciprocal_lattice_vectors = np.array([
            (
                h / self.lattice_parameter,
                k / self.lattice_parameter,
                l / self.lattice_parameter
            )
            for h, k, l in self.hkl_indices
        ])

    '''I defined grain_size as a method instead of an attribute so that in later polycrstalline calculations, I could simply re-rotate a reciprocal lattice and change the size repeatedly in order to calculate projections. Currently not being used.'''

    def randomize_grain_size(self):
        return np.random.normal(grain.size_average, np.sqrt(grain.size_variance))  #Placeholder
    
    def randomize_grain_strain(self):
        return np.random.normal(grain.strain_average, np.sqrt(grain.strain_variance))  #Placeholder

    def randomize_rotation(self): 
        theta = np.radians(np.random.uniform(0, 360))  # rotate around z
        phi   = np.radians(np.random.uniform(0, 360))  # rotate around x

        """Returns the combined rotation matrix for given angles."""

        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        
        Rx = np.array([
            [1,  0,           0          ],
            [0,  np.cos(phi), -np.sin(phi)],
            [0,  np.sin(phi),  np.cos(phi)]
        ])

        R =  Rx @ Rz  # Apply rotation around x first, then z
        
        self.reciprocal_lattice_vectors = np.array(self.reciprocal_lattice_vectors)        
        self.reciprocal_lattice_vectors = self.reciprocal_lattice_vectors @ R.T  # 