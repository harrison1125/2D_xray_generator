# detector.py
import numpy as np
import math
import matplotlib.pyplot as plt

class Detector:
    """
    A class that represents the detector onto which EwaldSphere reflections
    are projected. This version has been EDITED to:
      1. Store a 2D image (self.image) representing the detector plane.
      2. Add a Gaussian "blob" for each valid reflection rather than a single pixel.
    """
    def __init__(self, ewald_sphere, experiment, detector_width, detector_height, detector_distance, structure_factor_func):
        self.experiment = experiment
        self.detector_width = detector_width
        self.detector_height = detector_height
        self.detector_distance = detector_distance
        self.ewald_sphere = ewald_sphere
        self.structure_factor_func = structure_factor_func

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

        # Create a temporary image for the current grain
        grain_image = np.zeros_like(self.image)

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

                # Debugging: Check the projected point coordinates
                # print(f"Projected Point (mm): x={projected_point[1]}, y={projected_point[2]}")

                # Convert from mm coords (centered at 0,0) to image pixel indices
                scale_x = self.image.shape[1] / self.detector_width
                scale_y = self.image.shape[0] / self.detector_height
                col = int((projected_point[1] * scale_x) + self.image.shape[1] / 2)
                row = int((projected_point[2] * scale_y) + self.image.shape[0] / 2)

                # Debugging: Check the pixel indices
                # print(f"Pixel Indices: row={row}, col={col}")

                # Define Gaussian parameters
                sigma = 2  # Standard deviation (spread) in pixels
                amplitude = 10.0  # Peak intensity
                f = 1

                if self.structure_factor_func is not None and len(point) >= 6:
                    h, k, l = int(point[3]), int(point[4]), int(point[5])
                    try:
                        F_hkl = self.structure_factor_func(h, k, l, f)
                        amplitude *= abs((F_hkl)**2)  # Square modulus gives intensity
                    except Exception as e:
                        print(f"Structure factor error for (hkl)=({h},{k},{l}): {e}")
                        amplitude = 0.0
                # Debugging: Check the Gaussian parameters
                # print(f"Gaussian Parameters: sigma={sigma}, amplitude={amplitude}")

                # Create a local region (±3σ) around (row, col) to limit computation
                row_min = max(0, row - int(3 * sigma))
                row_max = min(self.image.shape[0], row + int(3 * sigma) + 1)
                col_min = max(0, col - int(3 * sigma))
                col_max = min(self.image.shape[1], col + int(3 * sigma) + 1)

                # Debugging: Check the local region bounds
                # print(f"Local Region Bounds: row_min={row_min}, row_max={row_max}, col_min={col_min}, col_max={col_max}")

                # Build a mesh over this local region
                x_indices = np.arange(row_min, row_max)
                y_indices = np.arange(col_min, col_max)
                yy, xx = np.meshgrid(y_indices, x_indices)

                # Calculate a 2D Gaussian centered at (row, col)
                gaussian = amplitude * np.exp(-((xx - row)**2 + (yy - col)**2) / (2 * sigma**2))

                '''
                # Debugging: Visualize the Gaussian spot
                print(f"Gaussian Spot Shape: {gaussian.shape}")
                plt.imshow(gaussian, cmap='hot', origin='lower')
                plt.colorbar()
                plt.title("Single Gaussian Spot")
                plt.show()
                '''

                # Accumulate the Gaussian intensities into the grain image
                grain_image[row_min:row_max, col_min:col_max] += gaussian

                '''
                # Debugging: Visualize the detector image after adding the Gaussian
                plt.imshow(grain_image, cmap='hot', origin='lower', extent=[-self.detector_width / 2, self.detector_width / 2, -self.detector_height / 2, self.detector_height / 2])
                plt.colorbar()
                plt.title("Detector Image After Adding Gaussian Spot")
                plt.xlabel("Detector Width (mm)")
                plt.ylabel("Detector Height (mm)")
                plt.show()
                '''

        '''
        # Debugging: Final detector image
        print("Final Detector Image:")
        plt.imshow(self.image, cmap='hot', origin='lower', extent=[-self.detector_width / 2, self.detector_width / 2, -self.detector_height / 2, self.detector_height / 2])
        plt.colorbar()
        plt.title("Final Detector Image with Gaussian Spots")
        plt.xlabel("Detector Width (mm)")
        plt.ylabel("Detector Height (mm)")
        plt.show()
        '''

        # Return the updated image in addition to the projected coordinates and two_theta
        return {
            "coordinate": np.array(self.projected_points),
            "two_theta": np.array(self.two_theta),
            "image": grain_image  # Return the grain image instead of self.image
            # trying out returning self.image instead of grain image
        }
