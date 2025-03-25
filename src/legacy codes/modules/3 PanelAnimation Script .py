'''
Script specifically built for animating the process of adding grains individually. This was built specifically for the HTMAX March 5 Review, where I wanted to show the process intuitively. 
'''
# Example Usage
num_grains = int(input('how many grains?'))
exp = Experiment(wavelength=0.154, sample="Arbitary")
grain = Grain(size_average=10, size_variance=4, strain_average=10, strain_variance=4, aspect_ratio=1.5, lattice_parameter=0.361, experiment=exp)
points_on_detector = []

for _ in range(num_grains):
    grain.randomize_rotation()
    grain.randomize_grain_size()
    grain.randomize_grain_strain()
    ewald = EwaldSphere(grain, exp, 0.03)
    filtered_points = ewald.filter_points()
    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300, filtered_points = filtered_points)
    projected_points = detector.project_points()
    points_on_detector.extend(projected_points["coordinate"])

points_on_detector = np.array(points_on_detector)


# Plot
if points_on_detector is not None and len(points_on_detector) > 0:
    projected_points = np.array(projected_points)
    x_coords = points_on_detector[:, 1]
    y_coords = points_on_detector[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axis
#    ax.set_facecolor('#d6fffe')  # Pale blue background for the graph (axes)
    plt.scatter(x_coords, y_coords, c='orange', s = 3, label="Projected Points")
    plt.xlim(-detector.detector_width / 2, detector.detector_width / 2)
    plt.ylim(-detector.detector_height / 2, detector.detector_height / 2)  
    plt.xlabel("Detector Width (mm)")
    plt.ylabel("Detector Height (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No projected points to display.")

x_coords = []
y_coords = []
for i in range(num_grains):
    grain.randomize_rotation()
    ewald = EwaldSphere(grain, exp, 0.03)
    reciprocal_lattice = grain.reciprocal_lattice_vectors
    filtered_points = ewald.filter_points()

    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(reciprocal_lattice[:, 0], reciprocal_lattice[:, 1], reciprocal_lattice[:, 2], c='b', s = 5)
    ax1.set_title("Reciprocal Space and Ewald Sphere")

    elevation = 0  # Elevation angle in degrees (vertical rotation)
    azimuth = 269# Azimuth angle in degrees (horizontal rotation)
    ax1.view_init(elev=elevation, azim=azimuth)
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(reciprocal_lattice[:, 0], reciprocal_lattice[:, 1], reciprocal_lattice[:, 2], c='b', s= 10)
    # Plot the detector rectangle
    detector_corners = np.array([
        [detector.detector_distance, -detector.detector_width / 2, -detector.detector_height / 2],
        [detector.detector_distance, detector.detector_width / 2, -detector.detector_height / 2],
        [detector.detector_distance, detector.detector_width / 2, detector.detector_height / 2],
        [detector.detector_distance, -detector.detector_width / 2, detector.detector_height / 2],
    ])
    detector_x = [corner[0] for corner in detector_corners] + [detector_corners[0][0]]
    detector_y = [corner[1] for corner in detector_corners] + [detector_corners[0][1]]
    detector_z = [corner[2] for corner in detector_corners] + [detector_corners[0][2]]
    ax2.plot(detector_x, detector_y, detector_z, color='red', label='Detector')

    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300, filtered_points=filtered_points)
    projected_points_dict = detector.project_points()
    projected_points = np.array(projected_points_dict["coordinate"]) if "coordinate" in projected_points_dict else np.array([])
    
    # Plot the projections on the detector
    if len(projected_points) > 0:
        ax2.scatter(detector.detector_distance, projected_points[:, 0], projected_points[:, 1], c='orange', label='Projections on Detector', s=30)

    # Draw lines from sphere center through filtered points to detector
    for point, proj in zip(filtered_points, projected_points):
        ax2.plot([ewald.center[0], point[0], proj[0]],
                [ewald.center[1], point[1], proj[1]],
                [ewald.center[2], point[2], proj[2]], color='gray', alpha=0.5)
    
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")
    ax2.set_title("Reciprocal Space Projection on Detector")
    ax2.legend()


    ax2.set_title("Reciprocal Space Projection on Detector")
    ax2.view_init(elev=elevation, azim=azimuth)

    detector_points = projected_points 

    if detector_points.shape != 0:
        if detector_points.ndim == 1:
            detector_points = detector_points.reshape(-1, 3)  # Assuming it should have 3 columns
        
        x_coords.extend(detector_points[:, 1])
        y_coords.extend(detector_points[:, 2])

    ax3 = fig.add_subplot(133)
    ax3.scatter(x_coords, y_coords, c='orange', s = 3, label="Projected Points")

    ax3.set_xlim(-detector.detector_width / 2, detector.detector_width / 2)

    ax3.set_ylim(-detector.detector_height / 2, detector.detector_height / 2)  
    ax3.set_title(f"2D Detector View: Grain {i+1}")
    ax3.set_aspect('equal')
    output_dir = os.path.expanduser("~/Desktop/grain_frames")
    os.makedirs(output_dir, exist_ok=True)
        # Save frame
    frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    plt.savefig(frame_path)
    plt.close(fig)

    plt.show()


# Create video from frames

video_path = os.path.expanduser("~/Desktop/grains_projection.mp4")
frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")])

# Get frame size
frame = cv2.imread(frame_files[0])
height, width, layers = frame.shape

# Use the highest FPS needed
fps = 1
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    out.write(frame)

out.release()
print(f"Video saved at: {video_path}")
