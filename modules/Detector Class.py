class Detector: 
    '''
    I'm alive: I'm a detector class
    I know where I am relative to the Ewald Sphere. I know what is being projected onto me. 
    I know how to calculate all the 2thetas on myself. 
    I can instrumentally broaden all the points on myself. I do, however, use an intermediary objecct, defined later (point_on_detector), to store all the information about the broadened point itself. Honestly, this secondary object could probably be done within the Detector object itself I am just careful to do so in case this object becomes a little too unwieldy. 
    Projected_Points).
    '''
    def __init__ (self, ewald_sphere, Experiment, detector_width, detector_height, detector_distance, filtered_points):
        self.experiment = Experiment
        self.detector_width = detector_width
        self.detector_height = detector_height
        self.detector_distance = detector_distance
        self.ewald_sphere = ewald_sphere
        self.filtered_points = filtered_points 
        detector_corners = np.array([
            [detector_distance, -detector_width / 2, -detector_height / 2],
            [detector_distance, detector_width / 2, -detector_height / 2],
            [detector_distance, detector_width / 2, detector_height / 2],
            [detector_distance, -detector_width / 2, detector_height / 2],
        ])
        self.detector_x = [corner[0] for corner in detector_corners] + [detector_corners[0][0]]
        self.detector_y = [corner[1] for corner in detector_corners] + [detector_corners[0][1]]
        self.detector_z = [corner[2] for corner in detector_corners] + [detector_corners[0][2]]

        self.projected_points = []
        self.two_theta = []

    def project_points(self):
        filtered_points = self.ewald_sphere.filter_points()
        for point in filtered_points:
            # Extract the necessary components.
            xyz_coords = np.array(point[:3])   # x, y, z coordinates

            # Calculate the direction vector from the sphere center to the point.
            direction_vector = xyz_coords - self.ewald_sphere.center

            # Calculate the scale factor so that the point projects onto the plane at x = detector_distance.
            scale_factor = (self.detector_distance - self.ewald_sphere.center[0]) / direction_vector[0]

            # Calculate the projected point.
            projected_point = np.array(self.ewald_sphere.center) + scale_factor * direction_vector
            # Check if the projected point falls within the detector boundaries (y and z).
            if (
                -self.detector_width / 2  <= projected_point[1] <= self.detector_width / 2 and
                -self.detector_height / 2 <= projected_point[2] <= self.detector_height / 2
            ):
                self.projected_points.append(projected_point)

                two_theta_angle = math.atan((((projected_point[1])**2 + projected_point[2]**2)**0.5)/self.detector_distance)
                if two_theta_angle != 0:

                    self.projected_points.append(projected_point)
                    self.two_theta.append(two_theta_angle)

            return {
                "coordinate": np.array(self.projected_points), #defines the location of the projected point on the detector itself
                "two_theta": np.array(self.two_theta) # Calculates the 2theta angle that the point makes on the detector. 
            #if I want to be a little unwieldy, I can likely define the entirety of point_on_detector, including all broadening operations, here as well.
                }

        return np.array(self.projected_points)
    
