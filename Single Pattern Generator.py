'''
Script for running a single diffraction pattern 
'''
num_grains = int(input('how many grains?'))
exp = Experiment(wavelength=0.154, sample="Arbitary")
grain = Grain(size_average=33500, size_variance=33062500, strain_average=0, strain_variance=0, aspect_ratio=1.5, lattice_parameter=0.361, experiment=exp)
coords_on_detector = []

for _ in range(num_grains):
    grain.randomize_rotation()
    grain.randomize_grain_size()
    grain.randomize_grain_strain()
    
    ewald = EwaldSphere(grain, exp, 0.03)
    filtered_points = ewald.filter_points()

    detector = Detector(ewald, exp, detector_width=1000, detector_height=1000, detector_distance=300, filtered_points = filtered_points)
    projected_points = detector.project_points()
    coords_on_detector.extend(projected_points["coordinate"])
coords_on_detector = np.array(coords_on_detector)


# Plot
if coords_on_detector is not None and len(coords_on_detector) > 0:
    projected_points = np.array(projected_points)
    x_coords = coords_on_detector[:, 1]
    y_coords = coords_on_detector[:, 2]
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
