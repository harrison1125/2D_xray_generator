# experiment.py
import numpy as np


class Experiment:
    """Handles experimental conditions such as wavelength and sample."""

    def __init__(self, wavelength: float, sample: str):
        self.wavelength = wavelength
        self.sample = sample
        self.inv_lambda = 1 / wavelength


class Detector:
    """Simulates a 2D detector capturing XRD patterns."""

    def __init__(self, width: int, height: int, distance: float):
        self.width = width
        self.height = height
        self.distance = distance
        self.projected_points = []

    def add_projected_points(self, points):
        """Stores projected points on the detector."""
        self.projected_points.extend(points)
