#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# ---- Load NPZ file ----
npz_path = "/Users/angusgray/Desktop/Dissertation/github_page/cpg_modeling_als/CPG_ALS_lab_git/saved_simulations/v0d_ramp_results.npz"
out_dir = os.path.dirname(npz_path)

data = np.load(npz_path, allow_pickle=True)

weights = data["weights"]
spikes_L = data["spikes_L"]      # L_MNP1
spikes_R = data["spikes_R"]      # R_MNP2

# ---- Parameters ----
bin_ms = 10
sim_time = 1000
bins = np.arange(0, sim_time + bin_ms, bin_ms)

# ---- Plot burst envelopes for each block ----
fig, axs = plt.subplots(len(weights), 1, figsize=(10, 3 * len(weights)), sharex=True)

for i, (w, spL, spR) in enumerate(zip(weights, spikes_L, spikes_R)):
    rate_L, _ = np.histogram(spL, bins=bins)
    rate_R, _ = np.histogram(spR, bins=bins)

    # Smooth activity to show burst envelopes instead of raw spiking
    rate_L = gaussian_filter1d(rate_L, 3)
    rate_R = gaussian_filter1d(rate_R, 3)

    t = bins[:-1]
    axs[i].plot(t, rate_L, label="L_MNP1 (flexor)")
    axs[i].plot(t, rate_R, label="R_MNP2 (extensor)")
    axs[i].set_ylabel(f"W={w:.2f}")
    axs[i].legend(fontsize="x-small")
    axs[i].grid(alpha=0.2)

axs[-1].set_xlabel("Time (ms)")
plt.suptitle("Motor Neuron Burst Pattern Across V0D Weight Ramp", fontsize=14)
plt.tight_layout()

# ---- Save figure ----
out_file = os.path.join(out_dir, "burst_patterns_vs_v0d_weight.png")
plt.savefig(out_file, dpi=300)
plt.close()

print(f"\n[✔] Burst-rate figure saved to:\n{out_file}\n")
