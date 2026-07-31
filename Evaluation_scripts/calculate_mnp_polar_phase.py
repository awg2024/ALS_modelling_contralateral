#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from calculate_v0d_phase_stability import calculate_peak_to_peak_phase_diag_exact

# How to run 
# python calculate_mnp_phase_stability .../saved_simulations/V0D_SILENCED? V0D_ACTIVE 

# PARAMETERS (match analysis)
TIME_RESOLUTION_MS = 0.1
MNP_PHASE_Y_LINE = 0.4 # burst threshold? 
MIN_DIST_PHASE_CALC = 1000

plt.rcParams.update({'font.size': 18})


# FIND RUNS
def find_lr_mnp1_pairs(root):
    """
    Finds (LEFT mnp1, RIGHT mnp1) pairs under all Figures folders.
    Returns list of (L_path, R_path).
    """
    pairs = []
    left_files = glob.glob(
        os.path.join(root, "**", "Figures", "LEFT_output_mnp1.csv"),
        recursive=True
    )

    for L in left_files:
        R = L.replace("LEFT_output_mnp1.csv", "RIGHT_output_mnp1.csv")
        if os.path.exists(R):
            pairs.append((L, R))

    return pairs

# POLAR PLOT
def plot_mnp1_phase_polar(phases_deg, label):
    phases_deg = np.asarray(phases_deg)
    phases_deg = phases_deg[np.isfinite(phases_deg)]

    if phases_deg.size == 0:
        print("[WARN] No valid phase values")
        return
    # Expand 0–180 → 0–360
    theta = np.deg2rad(phases_deg % 360) 

    # Unit radius (one point per simulation)
    r = np.ones_like(theta)

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")

    # Individual points
    ax.scatter(theta, r, s=80, alpha=0.7, label="Individual_Phase")

    # Mean direction
    mean_theta = np.angle(np.mean(np.exp(1j * theta)))
    if mean_theta < 0:
        mean_theta += 2 * np.pi

    #ax.scatter(mean_theta, 1, s=90, alpha=0.8, label="Mean_Phase")

    # Formatting
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
    )

    ax.set_title(f"MNP1 L–R Phase (0–360°)\n{label}", pad=20)
    ax.legend()

    plt.tight_layout()
    plt.show()


# MAIN
def main(root):
    pairs = find_lr_mnp1_pairs(root)
    print(f"[INFO] Found {len(pairs)} simulations")

    phases = []

    for L_path, R_path in pairs:
        L = np.loadtxt(L_path, delimiter=",")
        R = np.loadtxt(R_path, delimiter=",")

        # Normalize only for peak detection
        L_norm = (L - np.min(L)) / (np.max(L) - np.min(L) + 1e-12)
        R_norm = (R - np.min(R)) / (np.max(R) - np.min(R) + 1e-12)

        phase_deg, *_ = calculate_peak_to_peak_phase_diag_exact(
            L_norm, R_norm,
            bin_ms=TIME_RESOLUTION_MS,
            min_peak_height=MNP_PHASE_Y_LINE,
            min_dist_bins=MIN_DIST_PHASE_CALC,
            prominence=0.25
        )

        if np.isfinite(phase_deg):
            phases.append(phase_deg)

    plot_mnp1_phase_polar(phases, os.path.basename(os.path.abspath(root)))

# ENTRY POINT
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: plot_mnp1_lr_phase_polar.py <root_folder>")
        sys.exit(1)

    main(sys.argv[1])