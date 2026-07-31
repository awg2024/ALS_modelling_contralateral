#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
from calculate_stability_metrics_all_trials import calculate_peak_to_peak_phase   # ← ground-truth function
import sys
import time 
import pandas as pd
import re
import os

# PARAMETERS — match compare_metrics_across_timepoints_lr_rgmnp.py exactly
TIME_RESOLUTION_MS   = 0.1
MNP_PHASE_Y_LINE     = 0.4    # burst_threshold used in compare script (schema["burst_threshold"])
MIN_DIST_PHASE_CALC  = 1000   # min_peak_distance from POP_SCHEMAS["MNP"]

save_as_svg = 1 # if you want your figures to be saved as svg in the save_path below 
save_seed_csv = 1  # if you want a per-seed phase CSV saved (needed for the bio-vs-comp LMM)

# code to seperate out the MNP data and plot with different labels 
plot_phase_ablation = 0
ablation_window     = (0, 5000)      # ms — the silenced period
RECOVERY_END_MS = 10000   # add this near your other parameters at the top

SAVE_PATH = r"/mnt/d/Users/ag399/Dissertation/polar_svg_computational"


save_path_comparision = r"/mnt/d/Users/ag399/Dissertation/phase_comparision"

plt.rcParams.update({'font.size': 18})

CONDITION_COLOR_MAP = {
    "p0": "goldenrod",
    "p45": "hotpink",
    "p63": "mediumpurple",
    "p112": "lightskyblue",
}


# ===================================
# FIND LEFT/RIGHT MNP1 PAIRS
# ===================================
def find_lr_mnp1_pairs(root):
    print("Searching in:", root)
    pattern    = os.path.join(root, "**", "LEFT_output_mnp1.csv")

    folder = os.path.basename(root)
    timepoint = folder.split("_")[0] # p45, p112, p0 etc. 

    print("Pattern:", pattern)
    left_files = glob.glob(pattern, recursive=True)
    print(f"Found LEFT files: {left_files} for timepoint: {timepoint}")

    pairs = []
    for L in left_files:
        R = L.replace("LEFT_output_mnp1.csv", "RIGHT_output_mnp1.csv")
        print("Checking pair:")
        print("  L:", L)
        print("  R:", R)
        if os.path.exists(R):
            pairs.append((L, R))
    return pairs, timepoint

def get_condition_color(label):

    label = label.lower()

    for condition, color in CONDITION_COLOR_MAP.items():
        if condition in label:
            return color

    return "gray"
# ===================================
# PHASE CALCULATION — ground truth
# Matches compute_inter_phase() in compare script exactly.
# ===================================


def compute_inter_phase_with_times(left_norm, right_norm):
    """
    Identical to compute_inter_phase() but also returns the timestamp (ms)
    of the LEFT peak that anchors each cycle — needed for ablation windowing.

    Returns
    -------
    inter_phase_scalar : float
    per_cycle_phases   : 1-D array of phase values (degrees)
    cycle_times_ms     : 1-D array of LEFT-peak timestamps (ms), same length
    """
    phase_peak, _, _, freq1, freq2, pop1_peaks, pop2_peaks, cv_freq1, cv_freq2 = \
        calculate_peak_to_peak_phase(
            left_norm, right_norm,
            MNP_PHASE_Y_LINE,
            MIN_DIST_PHASE_CALC,
        )

    inter_phase_scalar = 360 - phase_peak if phase_peak > 180 else phase_peak

    per_cycle_phases = []
    cycle_times_ms   = []

    p1 = np.array(pop1_peaks, dtype=float)
    p2 = np.array(pop2_peaks, dtype=float)

    if len(p1) >= 2 and len(p2) >= 1:
        for k in range(len(p1) - 1):
            local_period = p1[k + 1] - p1[k]
            if local_period <= 0:
                continue
            diffs     = np.abs(p2 - p1[k])
            nearest   = p2[np.argmin(diffs)]
            time_diff = nearest - p1[k]
            phase_deg = (time_diff / local_period) * 360.0
            phase_deg = phase_deg % 360
            phase_deg = 360 - phase_deg if phase_deg > 180 else phase_deg

            per_cycle_phases.append(phase_deg)
            cycle_times_ms.append(p1[k] * TIME_RESOLUTION_MS)   # samples → ms

    per_cycle_phases = np.array(per_cycle_phases, dtype=float)
    cycle_times_ms   = np.array(cycle_times_ms,   dtype=float)

    mask             = np.isfinite(per_cycle_phases)
    return inter_phase_scalar, per_cycle_phases[mask], cycle_times_ms[mask]


def _rose_panel(ax, phases_deg, title, group_color, n_bins=18):
    """
    Draw a single rose-histogram panel onto an existing polar Axes.
    Reuses the same style as plot_save_histogram_polar().
    """
    if len(phases_deg) == 0:
        ax.set_title(f"{title}\n(no data)", pad=20, fontsize=18)
        return

    bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_width   = bin_edges[1] - bin_edges[0]
    counts, _   = np.histogram(np.deg2rad(phases_deg % 360), bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax.spines['polar'].set_visible(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(linewidth=1.2)
    ax.tick_params(axis='x', labelsize=16)

    ax.bar(bin_centers, counts, width=bin_width, bottom=0,
           color=group_color, alpha=0.75, edgecolor='white', linewidth=0.6, zorder=2)

    max_count = counts.max() if counts.max() > 0 else 1
    tick_vals = (np.array([0.25, 0.50, 0.75, 1.0]) * max_count).astype(int)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([str(v) for v in tick_vals], fontsize=10, color='grey')

    # grand-mean arrow
    mean_phi, R = circular_mean(phases_deg)

    # circular SD (degrees)
    circ_sd = np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else np.nan

    ax.annotate("", xytext=(0.0, 0.0), xy=(mean_phi, max_count * R),
                arrowprops=dict(color='black', lw=3, width=0.75, fill=True), zorder=5)

    mean_deg = float(np.degrees(mean_phi) % 360)

    ax.set_title(
        f"{title}\n"
        f"n={len(phases_deg)} cycles   "
        f"mean={mean_deg:.1f}°   "
        f"R={R:.3f}   "
        f"circular SD={circ_sd:.1f}°",
        pad=20,
        fontsize=18,
    )


def plot_ablation_polar(
    all_per_cycle_phases,
    all_cycle_times_ms,
    ablation_window,
    recovery_end_ms=RECOVERY_END_MS,
    label="MNP1",
    n_bins=18,
):
    """
    Three side-by-side rose histograms:
      Panel 1 — Pre-ablation  : t < ablation_window[0]
      Panel 2 — Ablation      : ablation_window[0] ≤ t < ablation_window[1]
      Panel 3 — Recovery      : ablation_window[1] ≤ t < recovery_end_ms

    Parameters
    ----------
    all_per_cycle_phases : list of 1-D arrays (one per seed)
    all_cycle_times_ms   : list of 1-D arrays (matching timestamps, one per seed)
    ablation_window      : (start_ms, end_ms)
    recovery_end_ms      : end of recovery period in ms
    label                : string used in filename / suptitle
    n_bins               : rose histogram bins (default 18 → 20° each)
    """
    abl_start, abl_end = ablation_window

    pre_phases      = []
    ablation_phases = []
    recovery_phases = []

    for phases, times in zip(all_per_cycle_phases, all_cycle_times_ms):
        if phases is None or len(phases) == 0:
            continue
        phases = np.array(phases, dtype=float)
        times  = np.array(times,  dtype=float)
        mask   = np.isfinite(phases)
        phases, times = phases[mask], times[mask]

        pre_phases.extend(     phases[times <  abl_start].tolist())
        ablation_phases.extend(phases[(times >= abl_start) & (times < abl_end)].tolist())
        recovery_phases.extend(phases[(times >= abl_end)   & (times < recovery_end_ms)].tolist())

    pre_phases      = np.array(pre_phases,      dtype=float)
    ablation_phases = np.array(ablation_phases, dtype=float)
    recovery_phases = np.array(recovery_phases, dtype=float)

    # ── three-panel figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 9),
                             subplot_kw=dict(projection='polar'))

    _rose_panel(axes[0], pre_phases,
                f"Pre-Ablation\n(t < {abl_start} ms)",
                group_color='steelblue', n_bins=n_bins)

    _rose_panel(axes[1], ablation_phases,
                f"Ablation\n({abl_start}–{abl_end} ms)",
                group_color='tomato', n_bins=n_bins)

    _rose_panel(axes[2], recovery_phases,
                f"Recovery\n({abl_end}–{recovery_end_ms} ms)",
                group_color='mediumseagreen', n_bins=n_bins)

    fig.suptitle(f"Inter-Phase Ablation Analysis — {label}", fontsize=24, y=1.02)
    plt.tight_layout()

    # ── save as SVG ──────────────────────────────────────────────────────────
    safe_label = label.replace(" ", "_")
    out_dir    = os.path.join(SAVE_PATH, "svg_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(out_dir, f"ablation_polar_{safe_label}.svg")
    plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
    print(f"[SAVED] {out_path}")
    plt.close()



def compute_inter_phase(left_norm, right_norm):
    """
    Compute per-cycle inter-phase values for one trial.

    Uses the same ground-truth function and wrapping as compute_inter_phase()
    in compare_metrics_across_timepoints_lr_rgmnp.py, but returns:
      - inter_phase_scalar : the scalar summary (360 - phase_peak if > 180)
      - per_cycle_phases   : 1-D array of per-cycle phase values (degrees),
                             one per matched LEFT/RIGHT peak pair, with the
                             same wrapping applied to each cycle individually.
    """
    phase_peak, phase_variance_peak, coeff_phase_variance_peak, \
        freq1, freq2, pop1_peaks, pop2_peaks, cv_freq1, cv_freq2 = \
        calculate_peak_to_peak_phase(
            left_norm, right_norm,
            MNP_PHASE_Y_LINE,
            MIN_DIST_PHASE_CALC,
        )

    # Scalar summary — matches compare script exactly
    inter_phase_scalar = 360 - phase_peak if phase_peak > 180 else phase_peak

    # Per-cycle phases from matched peak pairs
    per_cycle_phases = []
    p1 = np.array(pop1_peaks, dtype=float)
    p2 = np.array(pop2_peaks, dtype=float)

    if len(p1) >= 2 and len(p2) >= 1:
        for k in range(len(p1) - 1):
            local_period = p1[k + 1] - p1[k]
            if local_period <= 0:
                continue
            diffs   = np.abs(p2 - p1[k])
            nearest = p2[np.argmin(diffs)]
            time_diff = nearest - p1[k]
            phase_deg = (time_diff / local_period) * 360.0
            phase_deg = phase_deg % 360
            phase_deg = 360 - phase_deg if phase_deg > 180 else phase_deg
            per_cycle_phases.append(phase_deg)

    per_cycle_phases = np.array(per_cycle_phases, dtype=float)
    per_cycle_phases = per_cycle_phases[np.isfinite(per_cycle_phases)]

    return inter_phase_scalar, per_cycle_phases


# ===================================
# HELPER: circular mean
# ===================================
def circular_mean(phases_deg):
    """Return (mean_angle_rad, vector_strength_R) for an array of degree values."""
    theta        = np.deg2rad(np.asarray(phases_deg, dtype=float) % 360)
    complex_mean = np.mean(np.exp(1j * theta))
    R            = np.abs(complex_mean)
    mean_phi     = np.angle(complex_mean)
    if mean_phi < 0:
        mean_phi += 2 * np.pi
    return mean_phi, R


# ===================================
# POLAR PLOT
# ===================================
def plot_spike_level_phase_polar(phases_list, label, collective_plot=False):
    """
    phases_list : single 1-D array (collective_plot=False)
                  or list of 1-D arrays, one per seed (collective_plot=True)
    """
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="polar")
    cmap = plt.cm.viridis

    if not collective_plot:
        phases_deg = np.atleast_1d(phases_list).astype(float)
        phases_deg = phases_deg[np.isfinite(phases_deg)]
        if len(phases_deg) == 0:
            print("No valid phase data")
            return

        theta = np.deg2rad(phases_deg % 360)
        r     = 0.7 + 0.3 * np.random.rand(len(theta))
        ax.scatter(theta, r, alpha=0.5, s=100, color="tab:blue")

        complex_mean = np.mean(np.exp(1j * theta))
        R_val        = np.abs(complex_mean)
        mean_theta   = np.angle(complex_mean)
        if mean_theta < 0:
            mean_theta += 2 * np.pi

        ax.plot([mean_theta, mean_theta], [0, R_val],
                linewidth=4, color="black", label="Mean Phase")

        print("\n===== Single Trial Summary =====")
        print(f"Mean Phase       = {np.degrees(mean_theta):.2f}°")
        print(f"Vector Strength R = {R_val:.3f}")

    else:
        global_phase_pool = []
        for i, phases_deg in enumerate(phases_list):
            if phases_deg is None:
                continue
            phases_deg = np.atleast_1d(phases_deg).astype(float)
            phases_deg = phases_deg[np.isfinite(phases_deg)]
            if len(phases_deg) == 0:
                continue

            global_phase_pool.append(phases_deg)
            theta = np.deg2rad(phases_deg % 360)
            r     = 0.7 + 0.3 * np.random.rand(len(theta))
            color = cmap(i / max(len(phases_list), 1))
            ax.scatter(theta, r, s=100, alpha=0.5, color=color, label=f"Seed {i+1}")

        if global_phase_pool:
            concat       = np.concatenate(global_phase_pool)
            theta_global = np.deg2rad(concat % 360)
            global_mean  = np.mean(np.exp(1j * theta_global))
            R_global     = np.abs(global_mean)
            mean_theta_g = np.angle(global_mean)
            if mean_theta_g < 0:
                mean_theta_g += 2 * np.pi
            ax.plot([mean_theta_g, mean_theta_g], [0, R_global],
                    linewidth=5, color="black", label="Global Mean")

    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    title_prefix = "COLLECTIVE Phase" if collective_plot else "Single Trial Phase"
    ax.set_title(f"{title_prefix}\n{label}", pad=20)
    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.show()


# ===================================
# SUMMARY STATISTICS
# ===================================
def summary_statistics(all_per_cycle_phases, all_scalar_phases):
    """
    all_per_cycle_phases : list of 1-D arrays (per-cycle values per seed)
    all_scalar_phases    : list of scalars (one summary value per seed)
    """
    print("\n===== Per-Seed Summary =====")
    print(f"  {'Seed':<8}  {'Scalar (°)':>12}  {'Cycles':>8}  {'Mean cycles (°)':>18}  {'SD (°)':>8}")
    for i, (scalar, cycles) in enumerate(zip(all_scalar_phases, all_per_cycle_phases)):
        cycles = np.array(cycles, dtype=float) if cycles is not None else np.array([])
        cycles = cycles[np.isfinite(cycles)]
        if len(cycles) > 0:
            print(f"  Seed {i+1:<3}  {scalar:>12.2f}  {len(cycles):>8}  {np.mean(cycles):>18.2f}  {np.std(cycles):>8.2f}")
        else:
            print(f"  Seed {i+1:<3}  {scalar:>12.2f}  {'0':>8}  {'NaN':>18}  {'NaN':>8}")

    valid_arrays = [np.array(p, dtype=float) for p in all_per_cycle_phases
                    if p is not None and len(p) > 0]
    if not valid_arrays:
        print("[WARN] No per-cycle phase data for collective stats")
        return

    all_phases = np.concatenate(valid_arrays)
    all_phases = all_phases[np.isfinite(all_phases)]
    if len(all_phases) == 0:
        print("[WARN] No valid phase data after filtering")
        return

    theta        = np.deg2rad(all_phases % 360)
    complex_mean = np.mean(np.exp(1j * theta))
    R_val        = np.abs(complex_mean)
    mean_theta   = np.angle(complex_mean)
    if mean_theta < 0:
        mean_theta += 2 * np.pi

    circular_std = np.sqrt(-2 * np.log(R_val + 1e-12))
    cv_stat      = np.std(all_phases) / (np.mean(all_phases) + 1e-12)

    print("\n===== Collective Phase Statistics (all cycles, all seeds) =====")
    print(f"Total cycles      = {len(all_phases)}")
    print(f"Mean Phase        = {np.degrees(mean_theta):.2f}°")
    print(f"Vector Strength R = {R_val:.3f}")
    print(f"Circular Std      = {np.degrees(circular_std):.2f}°")
    print(f"Coefficient of Variation = {cv_stat:.3f}")


# ===================================
# MULTI-SEED COLLECTIVE POLAR PLOT
# ===================================
def plot_multi_seed_collective(all_per_cycle_phases, all_scalar_phases):
    """
    all_per_cycle_phases : list of 1-D arrays, one per seed — per-cycle phase values
    all_scalar_phases    : list of scalars, one per seed — the summary value
    """
    fig  = plt.figure(figsize=(8, 8))
    ax   = fig.add_subplot(111, projection="polar")
    cmap = plt.cm.viridis

    global_pool = []

    for i, cycles in enumerate(all_per_cycle_phases):
        if cycles is None or len(cycles) == 0:
            continue

        cycles = np.array(cycles, dtype=float)
        cycles = cycles[np.isfinite(cycles)]
        if len(cycles) == 0:
            continue

        global_pool.append(cycles)
        color = cmap(i / max(len(all_per_cycle_phases), 1))
        theta = np.deg2rad(cycles % 360)
        r     = 0.7 + 0.3 * np.random.rand(len(theta))
        ax.scatter(theta, r, s=60, alpha=0.5, color=color, label=f"Seed {i+1}")

    if global_pool:
        concat       = np.concatenate(global_pool)
        theta_all    = np.deg2rad(concat % 360)
        global_mean  = np.mean(np.exp(1j * theta_all))
        R_global     = np.abs(global_mean)
        mean_theta_g = np.angle(global_mean)
        if mean_theta_g < 0:
            mean_theta_g += 2 * np.pi
        ax.plot([mean_theta_g, mean_theta_g], [0, R_global],
                linewidth=5, color="black", label="Global Mean")

    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Multi-Seed Inter-Phase (LEFT vs RIGHT MNP1)\nDots = per-cycle, Arrow = seed mean",
                 pad=20, fontsize=16)
    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False, fontsize=14)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.show()


# ===================================
# ROSE HISTOGRAM — save only, no show
# ===================================
def plot_save_histogram_polar(
    all_per_cycle_phases,
    all_scalar_phases,
    label="MNP1",
    n_bins=36, # match the systematic 
    group_color=None,
    timepoint=None
):

    """
    Builds a rose histogram of all per-cycle inter-phase values pooled across
    seeds, adds a grand-mean arrow, then saves to SAVE_PATH without displaying.

    Parameters
    ----------
    all_per_cycle_phases : list of 1-D arrays (one per seed)
    all_scalar_phases    : list of scalars  (one per seed)
    label                : string used in the filename and title
    timepoint            : timepoint of degeneration 
    n_bins               : number of equal-width bins around the circle (default 18 → 20° each)
    group_color          : fill colour for the bars
    """
    # ── pool all valid cycles ────────────────────────────────────────────────
    valid_arrays = [
        np.array(p, dtype=float)
        for p in all_per_cycle_phases
        if p is not None and len(p) > 0
    ]
    if not valid_arrays:
        print("[WARN] plot_save_histogram_polar: no valid phase data — skipping.")
        return

    all_phases_cat = np.concatenate(valid_arrays)
    all_phases_cat = all_phases_cat[np.isfinite(all_phases_cat)]
    if len(all_phases_cat) == 0:
        print("[WARN] plot_save_histogram_polar: all phases are NaN — skipping.")
        return

    n_total = len(all_phases_cat)
    n_seeds = sum(1 for p in all_per_cycle_phases if p is not None and len(p) > 0)

    # ── rose histogram ───────────────────────────────────────────────────────
    bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_width   = bin_edges[1] - bin_edges[0]
    counts, _   = np.histogram(np.deg2rad(all_phases_cat % 360), bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig = plt.figure(figsize=(16, 12))
    ax  = fig.add_subplot(111, projection="polar")
    ax.spines['polar'].set_visible(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(linewidth=1.2)
    ax.tick_params(axis='x', labelsize=18)

    ax.bar(
        bin_centers,
        counts,
        width=bin_width,
        bottom=0,
        color=group_color,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )

    # Radial ticks showing raw counts
    max_count  = counts.max() if counts.max() > 0 else 1
    tick_fracs = np.array([0.25, 0.50, 0.75, 1.0])
    tick_vals  = (tick_fracs * max_count).astype(int)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([str(v) for v in tick_vals], fontsize=11, color="grey")

    # grand-mean arrow
    mean_phi, R = circular_mean(all_phases_cat)
    arrow_len   = max_count * R

    ax.annotate("", xytext=(0.0, 0.0), xy=(mean_phi, arrow_len),
                arrowprops=dict(color="black", lw=3, width=0.75, fill=True),
                zorder=5)

    mean_deg = float(np.degrees(mean_phi) % 360)
    circ_sd = np.sqrt(-2 * np.log(R))

    legend_handles = [
        mpatches.Patch(facecolor=group_color, alpha=0.75, edgecolor="white",
                       label=f"Stride count  (n bins = {n_bins})"),
        plt.Line2D([0], [0], color="black", linewidth=2.5,
                   label=f"Grand mean  {mean_deg:.1f}°  (R={R:.3f}) Circular SD={circ_sd:.3f}"),
    ]

    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.25, 0.5), frameon=False, fontsize=20)

    ax.set_title(
        f"Inter-Phase Rose Histogram — {label} timepoint: {timepoint} \n"
        f"n = {n_total} cycles  ({n_seeds} seeds)",
        pad=20, fontsize=25,
    )

    plt.tight_layout()
    

    # ── save ────────────────────────────────────────────────────────────────
    safe_label = label.replace(" ", "_")

    if save_as_svg == 1:
        out_dir  = os.path.join(SAVE_PATH, "svg_output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"histogram_polar_{timepoint}.svg")
        plt.savefig(out_path, format='svg', bbox_inches="tight")
        print(f"[SAVED] {out_path}")
    else:
        out_dir  = os.path.join(SAVE_PATH, "png_output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"histogram_polar_{timepoint}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {out_path}")

    plt.show()   # after saving, if you still want to display it
    plt.close()


# ===================================
# SEED-LEVEL CSV EXPORT — needed for bio-vs-comp LMM
# ===================================
def export_seed_summary_csv(all_scalar_phases, pairs, timepoint, save_path_comparision):
    """
    Save one row per seed with its ground-truth scalar inter-phase value
    (the same `inter_phase_scalar` computed by compute_inter_phase_with_times,
    i.e. calculate_peak_to_peak_phase's phase_peak, wrapped to <=180°).

    This is the granularity the bio-vs-comp LMM comparison script expects:
    one Phase value per seed (a seed has no repeated-measures structure,
    unlike a mouse with 3 DLC trials).
    """
    rows = []
    for (L_path, R_path), scalar in zip(pairs, all_scalar_phases):

        # Path structure:
        # .../P0_D1/9881035_2026-05-05-11-21-00/Figures/LEFT_output_mnp1.csv
        seed_folder = os.path.basename(os.path.dirname(os.path.dirname(L_path)))

        match = re.search(r'\d+', seed_folder)

        if match:
            seed_name = int(match.group())
        else:
            print(f"[WARNING] Could not extract seed from: {seed_folder}")
            continue

        rows.append({
            "timepoint": timepoint,
            "seed_id":   seed_name,
            "phase_deg": scalar,
        })

    if not rows:
        print("[CSV] No seeds to export — skipping seed summary CSV.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(save_path_comparision, exist_ok=True)
    out_path = os.path.join(save_path_comparision, f"computational_seed_summary_{timepoint}.csv")
    df.to_csv(out_path, index=False)
    print(f"[CSV] Saved seed-level summary → {out_path}")


# ===================================
# PHASE DRIFT PLOT
# ===================================
def plot_phase_drift_sims(all_per_cycle_phases, show_individual=True, show_mean=True):
    """
    all_per_cycle_phases : list of 1-D arrays (one per seed).
    """
    fig, ax  = plt.subplots(figsize=(12, 6))
    cmap     = plt.cm.viridis
    max_cycles = 0

    if show_individual:
        for i, phases in enumerate(all_per_cycle_phases):
            if phases is None or len(phases) == 0:
                continue
            phases = np.array(phases, dtype=float)
            phases = phases[np.isfinite(phases)]
            if len(phases) == 0:
                continue
            cycles     = np.arange(1, len(phases) + 1)
            max_cycles = max(max_cycles, len(phases))
            color      = cmap(i / max(len(all_per_cycle_phases), 1))
            ax.plot(cycles, phases, color=color, alpha=0.5, linewidth=1.2,
                    label=f"Seed {i+1}")
            ax.scatter(cycles, phases, color=color, s=30, alpha=0.8, zorder=3)

    if show_mean and len(all_per_cycle_phases) > 1:
        valid   = [np.array(p, dtype=float) for p in all_per_cycle_phases
                   if p is not None and len(p) > 0]
        valid   = [p[np.isfinite(p)] for p in valid if len(p) > 0]
        if len(valid) > 1:
            min_len = min(len(p) for p in valid)
            if min_len >= 2:
                stacked    = np.array([p[:min_len] for p in valid])
                mean_phase = np.mean(stacked, axis=0)
                std_phase  = np.std(stacked,  axis=0)
                cycles     = np.arange(1, min_len + 1)
                ax.plot(cycles, mean_phase,
                        color="black", linewidth=3, zorder=5, label="Mean")
                ax.fill_between(cycles,
                                mean_phase - std_phase,
                                mean_phase + std_phase,
                                color="black", alpha=0.15, label="±1 SD")

    ax.axhline(180, color="steelblue", linestyle="--",
               linewidth=2, alpha=0.8, label="180° (antiphase)")
    ax.axhline(0,   color="tomato",    linestyle="--",
               linewidth=2, alpha=0.8, label="0° (synchrony)")

    ax.set_xlabel("Cycle Number", fontsize=14)
    ax.set_ylabel("Inter-Phase (degrees)", fontsize=14)
    ax.set_title("Per-Cycle Phase Drift — Each Seed (LEFT vs RIGHT MNP1)",
                 fontsize=16, pad=15)
    ax.set_ylim(0, 200)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_yticklabels(["0°", "45°", "90°", "135°", "180°"])
    ax.set_xlim(1, max(max_cycles, 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()
    plt.show()


# ===================================
# MAIN
# ===================================
def main(root):
   
    all_scalar_phases    = []
    all_per_cycle_phases = []
    all_cycle_times_ms   = []   
    
    pairs, timepoint = find_lr_mnp1_pairs(root) # finding pairs based off root
    print(f"[INFO] Found {len(pairs)} simulations for timepoint: {timepoint}")

    for seed_id, (L_path, R_path) in enumerate(pairs):
       
        seed_folder = os.path.basename(os.path.dirname(os.path.dirname(L_path)))

        seed_name = [int(re.match(r'\d+', seed_folder).group())]


        print(f"\n===== Processing seed: {seed_name} =====")

        left_data  = np.loadtxt(L_path, delimiter=",", dtype=float)
        right_data = np.loadtxt(R_path, delimiter=",", dtype=float)

        left_norm  = (left_data  - left_data.min())  / (left_data.max()  - left_data.min()  + 1e-12)
        right_norm = (right_data - right_data.min()) / (right_data.max() - right_data.min() + 1e-12)

        try:
             #use the new time-aware version instead of compute_inter_phase()
            inter_phase_scalar, per_cycle_phases, cycle_times_ms = \
                compute_inter_phase_with_times(left_norm, right_norm)
        except Exception as e:
            print(f"  [INTER-PHASE ERROR] {e}")
            inter_phase_scalar = np.nan
            per_cycle_phases   = np.array([])
            cycle_times_ms     = np.array([])

        print(f"  Scalar={inter_phase_scalar:.2f}°  |  {len(per_cycle_phases)} cycles")
        all_scalar_phases.append(inter_phase_scalar)
        all_per_cycle_phases.append(per_cycle_phases)
        all_cycle_times_ms.append(cycle_times_ms)

    # ---- Figure 1: collective polar with per-cycle dots ----
    plot_multi_seed_collective(all_per_cycle_phases, all_scalar_phases)

    # ---- Figure 2: per-cycle drift across seeds ----
    plot_phase_drift_sims(all_per_cycle_phases)

    # ---- Figure 3: rose histogram — saved to disk, not displayed ----
    condition_color = get_condition_color(timepoint)

    plot_save_histogram_polar(all_per_cycle_phases, all_scalar_phases,
                            label="MNP1", group_color=condition_color, timepoint=timepoint)


    # ---- Summary stats ----
    summary_statistics(all_per_cycle_phases, all_scalar_phases)

    # ---- Seed-level CSV export (needed for the bio-vs-comp LMM) ----
    if save_seed_csv == 1:
        export_seed_summary_csv(all_scalar_phases, pairs, timepoint, save_path_comparision)

    if plot_phase_ablation:
        plot_ablation_polar(
            all_per_cycle_phases,
            all_cycle_times_ms,
            ablation_window=ablation_window,
            recovery_end_ms=RECOVERY_END_MS,
            label="MNP1",
        )


# ===================================
# ENTRY POINT
# ===================================
if __name__ == "__main__":
    print("[INFO] Running Inter-Phase Analysis (matched to compare script ground truth).")
    print("[USAGE] python calculate_intra_mnp_polar_phase.py <root_folder>")

    if len(sys.argv) != 2:
        print("Incorrect usage. Expected: python calculate_intra_mnp_polar_phase.py <root_folder>")
        sys.exit(1)

    main(sys.argv[1]) # this is the file path we are passing through