#!/Users/hpark108/miniconda3/envs/hexrdgui/bin/python

class Experiment:
    """Handles experimental conditions such as wavelength and sample."""

    def __init__(self, wavelength: float, sample: str):
        self.wavelength = wavelength
        self.sample = sample
        self.inv_lambda = 1 / wavelength


class Detector:
    '''
    I know where I am relative to the Ewald Sphere. 
    I know what is being projected onto me. 
    I know how to calculate all the 2thetas on myself. 
    I can instrumentally broaden all the points on myself. 
    I do, however, use an intermediary objecct, defined later (point_on_detector), 
    to store all the information about the broadened point itself. 
    Honestly, this secondary object could probably be done within the Detector object itself 
    I am just careful to do so in case this object becomes a little too unwieldy. 
    Projected_Points).
    '''
    def __init__(self, ewald_sphere, experiment, detector_width, detector_height, detector_distance):
        # EDIT: lowercase experiment, careful to not overwrite the class name
        self.experiment = experiment
        self.detector_width = detector_width
        self.detector_height = detector_height
        self.detector_distance = detector_distance
        self.ewald_sphere = ewald_sphere
        # EDIT: removed filtered points since I didn't see it used here, can add back if needed

        # Define detector corners
        detector_corners = np.array([
            [detector_distance, -detector_width / 2, -detector_height / 2],
            [detector_distance,  detector_width / 2, -detector_height / 2],
            [detector_distance,  detector_width / 2,  detector_height / 2],
            [detector_distance, -detector_width / 2,  detector_height / 2],
        ])

        # EDIT: vectorized the corner calculations with numpy slicing
        # faster and more readable
        self.detector_x = list(detector_corners[:, 0]) + [detector_corners[0, 0]]
        self.detector_y = list(detector_corners[:, 1]) + [detector_corners[0, 1]]
        self.detector_z = list(detector_corners[:, 2]) + [detector_corners[0, 2]]

        self.projected_points = []
        self.two_theta = []

    def project_points(self):
        filtered_points = self.ewald_sphere.filter_points()
        for point in filtered_points:
            # Extract the necessary components.
            xyz_coords = np.array(point[:3])

            # Calculate the direction vector from the sphere center to the point.
            direction_vector = xyz_coords - np.array(self.ewald_sphere.center)

            # Calculate the scale factor so that the point projects onto the plane at x = detector_distance.
            scale_factor = (self.detector_distance - self.ewald_sphere.center[0]) / direction_vector[0]

            # Calculate the projected point
            projected_point = np.array(self.ewald_sphere.center) + scale_factor * direction_vector

            # Check if the projected point is within the detector bounds (y and z)
            if (-self.detector_width / 2  <= projected_point[1] <= self.detector_width / 2 and
                -self.detector_height / 2 <= projected_point[2] <= self.detector_height / 2):
                self.projected_points.append(projected_point)
                two_theta_angle = math.atan(np.sqrt(projected_point[1]**2 + projected_point[2]**2) / self.detector_distance)
                if two_theta_angle != 0:
                    self.two_theta.append(two_theta_angle)

        # Edit: moved return statement outside the loop and deleted duplicate return
        return {
            "coordinate": np.array(self.projected_points),
            "two_theta": np.array(self.two_theta)
        }
