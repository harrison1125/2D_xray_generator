"""Matplotlib texture inspection: Bunge-ZXZ sections and equal-area pole figures."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .orientation import euler_to_quaternion, quaternion_to_rotation_matrix

def plot_odf_section(odf, phi2_deg=0, resolution=181, ax=None):
    """Plot ODF density on a Bunge ZXZ section (phi2 fixed; degrees)."""
    ax = plt.subplots()[1] if ax is None else ax
    phi1, Phi = np.meshgrid(np.linspace(0, 360, resolution), np.linspace(0, 180, resolution//2))
    angles = np.stack((phi1, Phi, np.full_like(phi1, phi2_deg)), axis=-1)
    image = ax.pcolormesh(phi1, Phi, odf.evaluate(euler_to_quaternion(angles, degrees=True)), shading="auto")
    ax.set(xlabel="$\\phi_1$ (deg)", ylabel="$\\Phi$ (deg)", title=f"ODF section, $\\phi_2$={phi2_deg}°")
    plt.colorbar(image, ax=ax, label="density / Haar measure")
    return ax

def plot_pole_figure(orientations, crystal_direction=(1,0,0), bins=80, ax=None, weights=None):
    """Equal-area upper-hemisphere pole figure of explicit grain quaternions."""
    ax = plt.subplots()[1] if ax is None else ax
    direction = np.asarray(crystal_direction, float); direction /= np.linalg.norm(direction)
    poles = np.einsum("nij,j->ni", quaternion_to_rotation_matrix(orientations), direction)
    poles[poles[:, 2] < 0] *= -1
    r = np.sqrt(2/(1 + poles[:, 2])); x, y = r*poles[:, 0], r*poles[:, 1]
    hist = ax.hist2d(x, y, bins=bins, range=[[-np.sqrt(2), np.sqrt(2)]]*2, weights=weights, cmap="magma")
    circle = plt.Circle((0, 0), np.sqrt(2), fill=False, color="white"); ax.add_patch(circle)
    ax.set_aspect("equal"); ax.set(xlabel="RD", ylabel="TD", title=f"Pole figure [{crystal_direction[0]}{crystal_direction[1]}{crystal_direction[2]}]")
    plt.colorbar(hist[3], ax=ax, label="weighted grain count")
    return ax
