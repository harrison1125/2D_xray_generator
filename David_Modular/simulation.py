# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from experiment import Experiment, Detector
from crystal import Grain
from ewald import EwaldSphere


def run_simulation(num_grains):
    """Runs the XRD simulation for a given number of grains."""
    experiment = Experiment(wavelength=0.154, sample="Arbitrary")
    detector = Detector(width=1000, height=1000, distance=300)

    for _ in range(num_grains):
        grain = Grain(
            size_avg=33500,
            size_var=33062500,
            strain_avg=0,
            strain_var=0,
            aspect_ratio=1.5,
            lattice_param=0.361,
            experiment=experiment,
        )
        grain.randomize_properties()
        ewald = EwaldSphere(grain, experiment)
        filtered_points = ewald.filter_points()
        detector.add_projected_points(filtered_points)

    return detector.projected_points


def plot_results(projected_points):
    """Plots the projected XRD points."""
    if projected_points:
        projected_points = np.array(projected_points)
        x_coords = projected_points[:, 1]
        y_coords = projected_points[:, 2]

        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, c="orange", s=3, label="Projected Points")
        plt.xlabel("Detector Width (mm)")
        plt.ylabel("Detector Height (mm)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No projected points to display.")


if __name__ == "__main__":
    num_grains = int(input("How many grains? "))
    projected_points = run_simulation(num_grains)
    plot_results(projected_points)
