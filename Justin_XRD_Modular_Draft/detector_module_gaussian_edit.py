# detector.py
import numpy as np
import math

class Detector:
    """
    A class that represents the detector onto which EwaldSphere reflections
    are projected. This version has been EDITED to:
      1. Store a 2D image (self.image) representing the detector plane.
      2. Add a Gaussian "blob" for each valid reflection rather than a single pixel.
    """
    def __init__(self, ewald_sphere, experiment, detector_width, detector_height, detector_distance):
        self.experiment = experiment
        self.detector_width = detector_width
        self.detector_height = detector_height
        self.detector_distance = detector_distance
        self.ewald_sphere = ewald_sphere

        # Detector corners (unchanged):
        detector_corners = np.array([
            [detector_distance, -detector_width / 2, -detector_height / 2],
            [detector_distance,  detector_width / 2, -detector_height / 2],
            [detector_distance,  detector_width / 2,  detector_height / 2],
            [detector_distance, -detector_width / 2,  detector_height / 2],
        ])

        self.detector_x = list(detector_corners[:, 0]) + [detector_corners[0, 0]]
        self.detector_y = list(detector_corners[:, 1]) + [detector_corners[0, 1]]
        self.detector_z = list(detector_corners[:, 2]) + [detector_corners[0, 2]]

        self.projected_points = []
        self.two_theta = []

        # ADDED: Initialize a 2D array (image) that will store Gaussian intensities.
        # We assume detector_width and detector_height represent pixel dimensions.
        self.image = np.zeros((int(detector_height), int(detector_width)))

    def project_points(self):
        """
        Projects valid reflection points from the Ewald sphere onto the detector plane.
        Each valid reflection is rendered as a 2D Gaussian in self.image.
        Returns a dict with:
            'coordinate': array of [x, y, z] projected points
            'two_theta': array of two-theta angles
            'image': the 2D detector image with Gaussian spots
        """
        filtered_points = self.ewald_sphere.filter_points()

        for point in filtered_points:
            xyz_coords = np.array(point[:3])
            direction_vector = xyz_coords - np.array(self.ewald_sphere.center)
            scale_factor = (self.detector_distance - self.ewald_sphere.center[0]) / direction_vector[0]
            projected_point = np.array(self.ewald_sphere.center) + scale_factor * direction_vector

            # Check if the projected point is within the detector's physical bounds:
            if (-self.detector_width / 2 <= projected_point[1] <= self.detector_width / 2 and
                -self.detector_height / 2 <= projected_point[2] <= self.detector_height / 2):

                self.projected_points.append(projected_point)

                two_theta_angle = math.atan(
                    np.sqrt(projected_point[1]**2 + projected_point[2]**2) / self.detector_distance
                )
                if two_theta_angle != 0:
                    self.two_theta.append(two_theta_angle)

                # ADDED: Convert from mm coords (centered at 0,0) to image pixel indices.
                # Detector center => pixel center (detector_width/2, detector_height/2)
                scale = 3 # ADDED HERE: Arbitrary scale factor for pixel conversion
                col = int((projected_point[1] * scale) + self.detector_width / 2)
                row = int((projected_point[2] * scale) + self.detector_height / 2)


                # ADDED: Define Gaussian parameters (can be set or imported from gauss_param.py)
                sigma = 5.0       # Standard deviation (spread) in pixels
                amplitude = 1.0   # Peak intensity

                # ADDED: Create a local region (±3σ) around (row, col) to limit computation
                row_min = max(0, row - int(3 * sigma))
                row_max = min(self.image.shape[0], row + int(3 * sigma) + 1)
                col_min = max(0, col - int(3 * sigma))
                col_max = min(self.image.shape[1], col + int(3 * sigma) + 1)

                # ADDED: Build a mesh over this local region
                x_indices = np.arange(row_min, row_max)
                y_indices = np.arange(col_min, col_max)
                yy, xx = np.meshgrid(y_indices, x_indices)

                # ADDED: Calculate a 2D Gaussian centered at (row, col)
                gaussian = amplitude * np.exp(-((xx - row)**2 + (yy - col)**2) / (2 * sigma**2))

                # ADDED: Accumulate the Gaussian intensities into self.image
                self.image[row_min:row_max, col_min:col_max] += gaussian

        # ADDED: Return the updated image in addition to the projected coordinates and two_theta
        return {
            "coordinate": np.array(self.projected_points),
            "two_theta": np.array(self.two_theta),
            "image": self.image
        }
