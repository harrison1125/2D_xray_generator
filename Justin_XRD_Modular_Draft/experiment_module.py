class Experiment:
    #consider moving the detector conditions to the detector object for code reusability ?
    def __init__(self, wavelength, sample): 
        self.wavelength = wavelength
        self.material = sample
        self.inv_lambda = 1 / wavelength
