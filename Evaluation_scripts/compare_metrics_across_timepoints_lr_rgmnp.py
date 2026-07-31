import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import shapiro
from scipy.stats import ttest_ind, kruskal, ttest_rel, wilcoxon, mannwhitneyu
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
import sys
import plotly.graph_objects as go
import plotly.subplots
import plotly.io as pio
import warnings
import pandas as pd
import re
import sys
from calculate_stability_metrics_all_trials import calculate_avg_peak, calculate_burst_duration_crossings, calculate_peak_to_peak_phase
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
from scipy.stats import mannwhitneyu
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import time 
from statsmodels.formula.api import mixedlm

# How to run:
# python compare_metrics_across_timepoints_lr_rgmnp.py /mnt/d/Users/ag399/Dissertation/ALS_modelling_contralateral/saved_simulations MNP P0_D1 P45_D1 P63_D1 P112_D1

DEBUG_SAVE = True

# Suppress all warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams['svg.fonttype'] = 'none'

# =========================================
# POPULATION SCHEMAS
# =========================================
POP_SCHEMAS = {
    "MNP": {
        "type":                 "paired",
        "files":                ["_output_mnp1.csv", "_output_mnp2.csv"],
        "num_neurons":          150,
        "has_phase":            True,
        "min_peak_distance":    1000,
        "burst_threshold":      0.4,
        "burst_threshold_raw":  8.5,
        "sampling_dt":          1.0,
        "time_resolution":      0.1,
    },
    "RG": {
        "type":                 "paired",
        "files":                ["_output_rg1.csv", "_output_rg2.csv"],
        "num_neurons":          200,
        "has_phase":            True,
        "min_peak_distance":    1000,
        "burst_threshold":      0.4,
        "burst_threshold_raw":  50,
        "sampling_dt":          1.0,
        "time_resolution":      0.1,
    },
}

# =========================================
# GLOBAL FLAGS
# =========================================
compare_to_healthy      = 1
save_as_svg             = 1
sanitycheck_parameters  = 0

# =========================================
# FIGURE STYLE
# =========================================
title_fontsize      = 24
axis_label_fontsize = 20
axis_line_thickness = 2
scatter_color_global = ['blue', 'orange', 'white']
group_spacing_global = 1.7
pair_offset_global   = 0.3

label_mapping = {
    "1_":  "",
    "2_":  " T2",
    "3_":  " T3",
    "4_":  " T4",
    "5_":  " T5",
    "6_":  " T6",
    "7_":  " T7",
    "8_":  " T8",
    "9_":  " T9",
    "10_": " T10",
    "11_": " T11",
    "12_": " T12",
    "13_": " T13",
}

METRIC_COLUMN_LABELS = [
    "Avg Max Neuron Firing Rate Flx",        # 0
    "Avg Max Neuron Firing Rate Ext",        # 1
    "Avg On-Cycle Neuron Firing Rate Flx",   # 2
    "Avg On-Cycle Neuron Firing Rate Ext",   # 3
    "Avg Off-Cycle Neuron Firing Rate Flx",  # 4
    "Avg Off-Cycle Neuron Firing Rate Ext",  # 5
    "Freq Flx",                              # 6
    "Freq Ext",                              # 7
    "Burst Duration Flx",                    # 8
    "Burst Duration Ext",                    # 9
    "Inter-Phase (Left MNP1 vs Right MNP1)", # 10
]

# =========================================
# CLEAN TERMINAL OUTPUT
# Only summary tables and p-values are printed.
# =========================================
def load_bio_phase(systemic_path):
    df = pd.read_csv(systemic_path)

    bio_data = {
        "healthy": df[(df["p_id"] == "p49") & (df["genotype"] == "WT")]["mean_phase_deg"].values,
        "p49":     df[(df["p_id"] == "p49") & (df["genotype"] == "SOD1")]["mean_phase_deg"].values,
        "p63":     df[(df["p_id"] == "p63") & (df["genotype"] == "SOD1")]["mean_phase_deg"].values,
        "p112":    df[(df["p_id"] == "p112") & (df["genotype"] == "SOD1")]["mean_phase_deg"].values,
    }

    return bio_data

def add_bio_comp_annotations(ax, positions, bio_data, comp_data, folders, pvals):
    y_max = ax.get_ylim()[1]

    idx = 0
    for tp in ["p49", "p63", "p112"]:
        p = pvals.get(tp, np.nan)

        if np.isnan(p):
            idx += 2
            continue

        x1 = positions[idx]
        x2 = positions[idx + 1]

        y = y_max * 0.9

        ax.plot([x1, x1, x2, x2],
                [y, y + 2, y + 2, y],
                color='black')

        if p < 0.001:
            s = "***"
        elif p < 0.01:
            s = "**"
        elif p < 0.05:
            s = "*"
        else:
            s = "ns"

        ax.text((x1 + x2) / 2, y + 3, s, ha='center')

        idx += 2

def compute_summary_table(bio_data, comp_data, folders):
    rows = []

    for tp in ["p49", "p63", "p112"]:
        comp_tag = TIMEPOINT_MAP[tp]

        comp_vals = []
        for f in folders:
            if comp_tag.lower() in f.lower():
                comp_vals = comp_data.get(f, [])
                break

        bio_vals = np.array(bio_data.get(tp, []), dtype=float)
        comp_vals = np.array(comp_vals, dtype=float)

        def stats(x):
            if len(x) == 0:
                return np.nan, np.nan, np.nan
            return np.mean(x), np.std(x, ddof=1), len(x)

        bio_mean, bio_sd, bio_n = stats(bio_vals)
        comp_mean, comp_sd, comp_n = stats(comp_vals)

        rows.append({
            "Timepoint": tp,
            "Bio mean": bio_mean,
            "Bio SD": bio_sd,
            "Bio n": bio_n,
            "Comp mean": comp_mean,
            "Comp SD": comp_sd,
            "Comp n": comp_n,
        })

    return pd.DataFrame(rows)


def plot_bio_vs_comp(ax, bio_data, comp_data, folders):
    positions = []
    data = []
    labels = []
    colors = []

    x = 0
    width = 0.3

    for tp in ["p49", "p63", "p112"]:
        comp_tag = TIMEPOINT_MAP[tp]

        comp_vals = []
        for f in folders:
            if comp_tag.lower() in f.lower():
                comp_vals = comp_data.get(f, [])
                break

        bio_vals = bio_data.get(tp, [])

        # positions
        positions.extend([x - width, x + width])
        data.extend([bio_vals, comp_vals])

        labels.append(tp)
        colors.extend(["#DD8452", "#55A868"])  # bio = orange, comp = green

        x += 1.5

    # boxplot
    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True)

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    # scatter
    for d, pos, c in zip(data, positions, colors):
        jitter = np.random.uniform(-0.05, 0.05, len(d))
        ax.scatter(np.full(len(d), pos) + jitter, d,
                   color=c, edgecolor='black', alpha=0.7)

    # xticks (centered)
    ax.set_xticks(np.arange(0, len(labels) * 1.5, 1.5))
    ax.set_xticklabels(labels)

    ax.set_ylabel("Inter-phase (°)")
    ax.set_title("Biological vs Computational Phase")

    df = compute_summary_table(bio_data, comp_data, folders)
    print(df.round(2))

    return positions



def compute_bio_vs_comp(bio_data, comp_data, folders):
    results = {}

    for bio_tp, comp_tag in TIMEPOINT_MAP.items():
        comp_vals = []
        for f in folders:
            if comp_tag.lower() in f.lower():
                comp_vals = comp_data.get(f, [])
                break

        bio_vals = bio_data.get(bio_tp, [])

        if len(bio_vals) > 1 and len(comp_vals) > 1:
            _, p = mannwhitneyu(bio_vals, comp_vals, alternative="two-sided")
        else:
            p = np.nan

        results[bio_tp] = p

    return results


def _collapse_to_bio_replicate(values, group_size=None):
    """
    Converts technical replicates → biological replicate means.
    Assumes values are already grouped per animal/session.
    """

    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.array([])

    # OPTION A (best): if you know grouping exists externally
    if group_size:
        return np.mean(values.reshape(-1, group_size), axis=1)

    # OPTION B (safe fallback): treat whole set as 1 biological replicate
    return np.array([np.mean(values)])



def run_statistics_wilcoxon_flx_ext(extracted_values_dict, folders, stats_dir, side, compare_trials):
    """
    Wilcoxon signed-rank test between flexor and extensor at each timepoint separately.
    Returns dict: {folder: {metric_name: p_value}}
    """
    metric_pairs = [
        (0, 1, "Max Firing Rate"),
        (2, 3, "On-Cycle Firing Rate"),
        (4, 5, "Off-Cycle Firing Rate"),
        (6, 7, "Frequency"),
        (8, 9, "Burst Duration"),
    ]

    results = {}
    rows    = []

    for folder in folders:
        label = folder.split('/')[-1].replace('P0', 'Healthy')
        results[folder] = {}
        data = np.array(extracted_values_dict.get(folder, []), dtype=float)
        if data.size == 0:
            continue

        for flx_idx, ext_idx, metric_name in metric_pairs:
            flx_vals = data[:, flx_idx]
            ext_vals = data[:, ext_idx]
            # Keep only rows where both are finite
            mask = np.isfinite(flx_vals) & np.isfinite(ext_vals)
            flx_clean = flx_vals[mask]
            ext_clean = ext_vals[mask]

            if len(flx_clean) < 3:
                p = np.nan
            else:
                try:
                    _, p = wilcoxon(flx_clean, ext_clean)
                except Exception:
                    p = np.nan

            results[folder][metric_name] = p
            rows.append([label, metric_name, f"{p:.4e}" if not np.isnan(p) else "NaN"])

    # Save to CSV
    try:
        csv_path = os.path.join(stats_dir, f"wilcoxon_flx_ext_{side}_{compare_trials}.csv")
        pd.DataFrame(rows, columns=["Condition", "Metric", "p_value"]).to_csv(csv_path, index=False)
    except Exception:
        pass

    return results

def _iqr_filter(values, k=2.5):
    """Return values within k*IQR of the median. Safe for small n."""
    if len(values) < 4:
        return values
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr < 1e-6:          # all values identical — no filtering needed
        return arr.tolist()
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)].tolist()

def _section(title, width=80):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def _subsection(title, width=60):
    print(f"\n  {'─' * width}")
    print(f"  {title}")
    print(f"  {'─' * width}")

def _print_summary_table(label, data_array):
    """
    Print mean ± SD / n for key metrics only.
    """
    if len(data_array) == 0:
        print(f"    {label:<20}  — no data")
        return
    arr  = np.array(data_array, dtype=float)
    n    = arr.shape[0]
    cols_to_show = [
        (0,  "Max FR Flx"),
        (1,  "Max FR Ext"),
        (6,  "Freq Flx (Hz)"),
        (7,  "Freq Ext (Hz)"),
        (8,  "Burst Dur Flx (ms)"),
        (9,  "Burst Dur Ext (ms)"),
    ]
    print(f"    {label}  (n={n})")
    print(f"      {'Metric':<24}  {'Mean':>8}  {'SD':>8}")
    for idx, name in cols_to_show:
        col = arr[:, idx]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            print(f"      {name:<24}  {'NaN':>8}  {'NaN':>8}")
        else:
            print(f"      {name:<24}  {np.mean(col):>8.3f}  {np.std(col):>8.3f}")

def _print_pvalue_table(side, folders, significance_results, metric_label_pairs,
                        kruskal_results=None, wilcoxon_flx_ext=None):
    folder_labels = [f.split('/')[-1].replace('P0', 'Healthy') for f in folders]
    sig_map = {'0.001': '***', '0.01': '**', '0.05': '*'}

    # ── Kruskal-Wallis summary ───────────────────────────────────────────
    if kruskal_results:
        _subsection(f"KRUSKAL-WALLIS  ▸  {side}")
        print(f"    {'Metric':<34}  {'H':>10}  {'p-value':>12}  {'Sig':>5}")
        print(f"    {'-'*34}  {'-'*10}  {'-'*12}  {'-'*5}")
        for col_label, display_name in metric_label_pairs:
            if col_label not in kruskal_results:
                continue
            H   = kruskal_results[col_label]["H"]
            p   = kruskal_results[col_label]["p"]
            stars = next((s for t, s in sig_map.items()
                          if not np.isnan(p) and p <= float(t)), "ns")
            p_str = f"{p:.4e}" if not np.isnan(p) else "NaN"
            H_str = f"{H:.3f}"  if not np.isnan(H) else "NaN"
            print(f"    {display_name:<34}  {H_str:>10}  {p_str:>12}  {stars:>5}")

    # ── Dunn post-hoc pairwise ───────────────────────────────────────────
    _subsection(f"DUNN POST-HOC (Bonferroni)  ▸  {side}")
    print(f"    {'Metric':<28}  {'Comparison':<36}  {'p-value':>12}  {'Sig':>5}")
    print(f"    {'-'*28}  {'-'*36}  {'-'*12}  {'-'*5}")
    for col_label, display_name in metric_label_pairs:
        if col_label not in significance_results:
            continue
        df = significance_results[col_label]
        n  = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    p = float(df.iloc[i, j])
                except Exception:
                    p = np.nan
                stars = next((s for t, s in sig_map.items()
                              if not np.isnan(p) and p <= float(t)), "ns")
                comp  = f"{folder_labels[i]} vs {folder_labels[j]}"
                p_str = f"{p:.4e}" if not np.isnan(p) else "NaN"
                print(f"    {display_name:<28}  {comp:<36}  {p_str:>12}  {stars:>5}")

    # ── Wilcoxon Flx vs Ext per timepoint ───────────────────────────────
    if wilcoxon_flx_ext:
        _subsection(f"WILCOXON SIGNED-RANK (Flx vs Ext per timepoint)  ▸  {side}")
        print(f"    {'Condition':<20}  {'Metric':<28}  {'p-value':>12}  {'Sig':>5}")
        print(f"    {'-'*20}  {'-'*28}  {'-'*12}  {'-'*5}")
        metric_display = {
            "Max Firing Rate":      "Max Firing Rate",
            "On-Cycle Firing Rate": "On-Cycle FR",
            "Off-Cycle Firing Rate":"Off-Cycle FR",
            "Frequency":            "Frequency",
            "Burst Duration":       "Burst Duration",
        }
        for folder in folders:
            cond_label = folder.split('/')[-1].replace('P0', 'Healthy')
            if folder not in wilcoxon_flx_ext:
                continue
            for metric_name, p in wilcoxon_flx_ext[folder].items():
                stars = next((s for t, s in sig_map.items()
                              if not np.isnan(p) and p <= float(t)), "ns")
                p_str = f"{p:.4e}" if not np.isnan(p) else "NaN"
                disp  = metric_display.get(metric_name, metric_name)
                print(f"    {cond_label:<20}  {disp:<28}  {p_str:>12}  {stars:>5}")

# =========================================
# ANALYSIS FUNCTIONS
# =========================================

def analyze_output(input_1, input_2, pop_type, y_line_bd, y_line_phase, min_dist, num_motor_neurons):
    try:
        pop_data1 = np.loadtxt(input_1, delimiter=',', dtype=float)
        pop_data2 = np.loadtxt(input_2, delimiter=',', dtype=float)

        avg_max_spike_rate_pop1, avg_max_spike_rate_pop2 = calculate_avg_peak(
            pop_data1, pop_data2, y_line_phase, min_dist
        )

        pop_data1 = [x / num_motor_neurons for x in pop_data1]
        pop_data2 = [x / num_motor_neurons for x in pop_data2]

        avg_duration_flx, avg_duration_ext, avg_on_cycle_flx, avg_off_cycle_flx, \
            avg_on_cycle_ext, avg_off_cycle_ext, avg_frequency_flx, avg_frequency_ext, \
            avg_phase = calculate_burst_duration_crossings(pop_data1, pop_data2)


        # print(f"avg frequency flx: {avg_frequency_flx}")
        # print(f"avg frequency ext: {avg_frequency_ext}")
        # time.sleep(50)



    except FileNotFoundError as e:
        print(f"  [ERROR] File not found: {e}")
        return (np.nan,) * 11
    except Exception as e:
        print(f"  [ERROR] Skipping trial: {e}")
        return (np.nan,) * 11

    return (
        avg_max_spike_rate_pop1,
        avg_max_spike_rate_pop2,
        avg_on_cycle_flx,
        avg_off_cycle_flx,
        avg_on_cycle_ext,
        avg_off_cycle_ext,
        avg_frequency_flx,
        avg_frequency_ext,
        avg_duration_flx,
        avg_duration_ext,
        np.nan,   # col 10 — inter-phase placeholder
    )

def compute_intra_phase_per_cycle(flx_file, ext_file, pop_type):
    """
    Compute per-cycle intra-phase (flexor vs extensor) for one hemicord trial.
 
    Parameters
    ----------
    flx_file : str   path to _output_mnp1.csv  (flexor population)
    ext_file : str   path to _output_mnp2.csv  (extensor population)
    pop_type : str   key into POP_SCHEMAS
 
    Returns
    -------
    cycle_phases : list[float]
        Per-cycle phase values in degrees (0–180).  Empty list on failure.
    mean_phase   : float   circular mean of cycle_phases, or np.nan
    """
    schema    = POP_SCHEMAS[pop_type]
    y_line    = schema["burst_threshold"]
    min_dist  = schema["min_peak_distance"]
 
    try:
        raw1 = np.loadtxt(flx_file, delimiter=',', dtype=float)
        raw2 = np.loadtxt(ext_file, delimiter=',', dtype=float)
    except Exception as e:
        print(f"  [INTRA-PHASE LOAD ERROR] {e}")
        return [], np.nan
 
    # Normalise to [0, 1]
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)
 
    pop1 = _norm(raw1)
    pop2 = _norm(raw2)
 
    try:
        # calculate_peak_to_peak_phase returns:
        #   phase_peak, phase_variance, coeff_variance,
        #   freq1, freq2, pop1_peaks, pop2_peaks, cv_freq1, cv_freq2
        (phase_peak, phase_variance, coeff_variance,
         freq1, freq2, pop1_peaks, pop2_peaks,
         cv_freq1, cv_freq2) = calculate_peak_to_peak_phase(
            pop1, pop2, y_line, min_dist
        )
    except Exception as e:
        print(f"  [INTRA-PHASE CALC ERROR] {e}")
        return [], np.nan
 
    if len(pop1_peaks) < 2 or len(pop2_peaks) < 2:
        return [], np.nan
 
    # ── Per-cycle phase values ───────────────────────────────────────────
    # Pair each pop1 (flx) peak with the nearest following pop2 (ext) peak
    # to get individual cycle phase estimates, then fold to 0-180°.
    cycle_phases = []
    pop2_arr = np.array(pop2_peaks)
 
    for p1 in pop1_peaks:
        # nearest ext peak AFTER this flx peak
        ahead = pop2_arr[pop2_arr > p1]
        if len(ahead) == 0:
            continue
        dt_samples = ahead[0] - p1
        # convert sample offset → fraction of cycle, then → degrees
        # use average inter-flx-peak interval as cycle length estimate
        if len(pop1_peaks) >= 2:
            cycle_len = np.mean(np.diff(pop1_peaks))
        else:
            continue
        if cycle_len < 1:
            continue
        phase_deg = (dt_samples / cycle_len) * 360.0
        phase_deg = 360 - phase_deg if phase_deg > 180 else phase_deg
        cycle_phases.append(float(phase_deg))
 
    if not cycle_phases:
        return [], np.nan
 
    # Circular mean
    rads      = np.deg2rad(cycle_phases)
    mean_rad  = np.angle(np.mean(np.exp(1j * rads))) % (2 * np.pi)
    mean_deg  = float(np.degrees(mean_rad))
    mean_deg  = 360 - mean_deg if mean_deg > 180 else mean_deg
 
    return cycle_phases, mean_deg

def plot_intra_phase_hemicord(intra_phase_dict, folders, output_dir, pop_type, stats_dir):
    n_folders     = len(folders)
    sides         = ["LEFT", "RIGHT"]
    side_colors   = {"LEFT": "#4878CF", "RIGHT": "#DD8452"}
    side_labels   = {"LEFT": "Left Hemicord", "RIGHT": "Right Hemicord"}
    sig_map       = {'0.001': '***', '0.01': '**', '0.05': '*'}
    MAX_SCATTER   = 50  # max points shown per group

    folder_labels = [
        f.split('/')[-1].replace('P0', 'Healthy').replace('_D1', '').replace('_D2', '')
        for f in folders
    ]

    # ── Figure: single panel, both sides overlaid ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Intra-Phase per Timepoint  —  {pop_type}",
        fontsize=13, fontweight='bold',
    )

    group_spacing = 1.6   # distance between timepoint clusters
    pair_offset   = 0.22  # LEFT/RIGHT offset within cluster

    side_offsets = {"LEFT": -pair_offset, "RIGHT": +pair_offset}

    all_x_centers = []  # for xticks
    y_top_global  = 0

    for tp_idx, (folder, tp_label) in enumerate(zip(folders, folder_labels)):
        x_center = 1 + tp_idx * group_spacing
        all_x_centers.append(x_center)

        for side in sides:
            color  = side_colors[side]
            x_pos  = x_center + side_offsets[side]
            vals   = [v for v in intra_phase_dict.get(side, {}).get(folder, [])
                      if not np.isnan(v)]

            if not vals:
                continue

            # Boxplot
            bp = ax.boxplot(
                [vals],
                positions=[x_pos],
                widths=pair_offset * 1.3,
                patch_artist=True,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker='', markersize=0),  # hide fliers; we draw scatter
            )
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.65)

            # Capped scatter — random subsample if too many points
            plot_vals = vals
            if len(vals) > MAX_SCATTER:
                rng       = np.random.default_rng(seed=42)
                plot_vals = rng.choice(vals, MAX_SCATTER, replace=False).tolist()

            jitter = np.random.uniform(-pair_offset * 0.35, pair_offset * 0.35, len(plot_vals))
            ax.scatter(
                np.full(len(plot_vals), x_pos) + jitter,
                plot_vals,
                color=color, edgecolors='black',
                alpha=0.55, s=20, zorder=3,
            )

            y_top_global = max(y_top_global, max(vals))

    # ── KW + Dunn brackets — run per side, stagger bracket height ────────
    bracket_level = 0
    step  = y_top_global * 0.09
    gap   = y_top_global * 0.02

    for side in sides:
        color = side_colors[side]
        groups_vals = []
        groups_xpos = []

        for tp_idx, folder in enumerate(folders):
            x_center = 1 + tp_idx * group_spacing
            x_pos    = x_center + side_offsets[side]
            vals     = [v for v in intra_phase_dict.get(side, {}).get(folder, [])
                        if not np.isnan(v)]
            groups_vals.append(vals)
            groups_xpos.append(x_pos)

        clean_vals = [np.array(d) for d in groups_vals if len(d) >= 3]
        clean_xpos = [xp for d, xp in zip(groups_vals, groups_xpos) if len(d) >= 3]

        if len(clean_vals) < 2:
            continue

        # KW annotation in xlabel area — print to console instead to avoid clutter
        try:
            from scipy.stats import kruskal as _kruskal
            H_kw, p_kw = _kruskal(*clean_vals)
            print(f"  [KW {side}]  H={H_kw:.2f}  p={p_kw:.4f}")
        except Exception:
            pass

        try:
            dunn   = sp.posthoc_dunn(clean_vals, p_adjust='bonferroni')
            n_grps = len(clean_vals)
            for i in range(n_grps):
                for j in range(i + 1, n_grps):
                    try:
                        p = float(dunn.iloc[i, j])
                    except Exception:
                        continue
                    stars = next((s for t, s in sig_map.items() if p <= float(t)), None)
                    if stars is None:
                        continue
                    x1, x2 = clean_xpos[i], clean_xpos[j]
                    y_br   = y_top_global + step * (bracket_level + 1)
                    ax.plot(
                        [x1, x1, x2, x2],
                        [y_br - gap, y_br, y_br, y_br - gap],
                        color=color, linewidth=1.2,
                    )
                    ax.text(
                        (x1 + x2) / 2, y_br + gap * 0.5,
                        stars, ha='center', va='bottom', fontsize=11, color=color,
                    )
                    bracket_level += 1
        except Exception:
            pass

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks(all_x_centers)
    ax.set_xticklabels(folder_labels, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel("Intra-Phase (°)", fontsize=12)
    ax.set_ylim(bottom=0, top=y_top_global + step * (bracket_level + 2))
    ax.tick_params(axis='y', labelsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)

    # ── Legend ────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=side_colors[s], alpha=0.65, edgecolor='black', label=side_labels[s])
        for s in sides
    ]
    ax.legend(handles=legend_handles, fontsize=11, framealpha=0.8, loc='upper right')

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    out_svg = os.path.join(output_dir, f"intra_phase_boxplot_{pop_type}.svg")
    out_png = os.path.join(output_dir, f"intra_phase_boxplot_{pop_type}.png")

    if save_as_svg == 1:
        fig.savefig(out_svg, format='svg', bbox_inches='tight', transparent=True)
        print(f"  Saved → {out_svg}")
    else:
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"  Saved → {out_png}")
    plt.close(fig)
 

def compute_inter_phase(left_file_1, right_file_1, pop_type, y_line_phase, min_dist):
    try:
        left_data  = np.loadtxt(left_file_1,  delimiter=',', dtype=float)
        right_data = np.loadtxt(right_file_1, delimiter=',', dtype=float)

        left_norm  = (left_data  - left_data.min())  / (left_data.max()  - left_data.min()  + 1e-12)
        right_norm = (right_data - right_data.min()) / (right_data.max() - right_data.min() + 1e-12)

        phase_peak, phase_variance_peak, coeff_phase_variance_peak, \
            freq1, freq2, pop1_peaks, pop2_peaks, cv_freq1, cv_freq2 = \
            calculate_peak_to_peak_phase(left_norm, right_norm, y_line_phase, min_dist)

        inter_phase = 360 - phase_peak if phase_peak > 180 else phase_peak
        return inter_phase

    except Exception as e:
        print(f"  [INTER-PHASE ERROR] {e}")
        return np.nan


def analyze_individual_population(input_file, pop_type, y_line_bd=0.4, min_dist=1000, num_neurons=150):
    try:
        pop_data_raw = np.loadtxt(input_file, delimiter=',', dtype=float)
        avg_max_rate_raw, _ = calculate_avg_peak(pop_data_raw, pop_data_raw, y_line_bd, min_dist)
        avg_max_rate = avg_max_rate_raw / num_neurons
        pop_data     = pop_data_raw / num_neurons
        (avg_duration, _, avg_on_cycle, avg_off_cycle, _, _, avg_frequency, _) = \
            calculate_burst_duration_crossings(pop_data, pop_data)
    except Exception as e:
        print(f"  [ERROR] Skipping {pop_type}: {e}")
        return (np.nan,) * 11
    return (avg_max_rate, np.nan, avg_on_cycle, avg_off_cycle,
            np.nan, np.nan, avg_frequency, np.nan, avg_duration, np.nan, np.nan)


def find_population_files(folder_containing_data, folder, pop_type, side):
    import glob
    pop_files         = POP_SCHEMAS[pop_type]["files"]
    search_patterns   = [
        os.path.join(folder_containing_data, folder, '**', f'{side}{fname}')
        for fname in pop_files
    ]
    matched_files_all = [glob.glob(p, recursive=True) for p in search_patterns]
    if not all(matched_files_all):
        print(f"  [WARNING] Not all files found for {pop_type} | {folder} | {side}")
        return []
    results = []
    for files_set in zip(*matched_files_all):
        trial_subfolder = os.path.relpath(
            os.path.dirname(files_set[0]),
            os.path.join(folder_containing_data, folder)
        )
        results.append((list(files_set), trial_subfolder))
    return results


def find_inter_phase_file_pairs(folder_containing_data, folder, pop_type):
    import glob
    pop_file_1    = POP_SCHEMAS[pop_type]["files"][0]
    left_pattern  = os.path.join(folder_containing_data, folder, '**', f'LEFT{pop_file_1}')
    right_pattern = os.path.join(folder_containing_data, folder, '**', f'RIGHT{pop_file_1}')
    left_matches  = sorted(glob.glob(left_pattern,  recursive=True))
    right_matches = sorted(glob.glob(right_pattern, recursive=True))
    if not left_matches or not right_matches:
        print(f"  [INTER-PHASE] No LEFT/RIGHT pairs found in {folder}")
        return []
    if len(left_matches) != len(right_matches):
        print(f"  [INTER-PHASE WARNING] Unequal LEFT/RIGHT counts in {folder}")
    results = []
    for left_f, right_f in zip(left_matches, right_matches):
        trial_subfolder = os.path.relpath(
            os.path.dirname(left_f),
            os.path.join(folder_containing_data, folder)
        )
        results.append((left_f, right_f, trial_subfolder))
    return results


def remove_outliers(trial_type, data_array, stats_dir):
    """
    3-SD outlier removal on cols 0-9. Returns full 11-column numeric array.
    """
    percentile_outliers = 0.0
    if len(data_array) == 0:
        return []

    data_array = np.array(data_array)
    if data_array.ndim != 2 or data_array.shape[0] == 0:
        return np.array([])

    numeric_data = data_array[:, :-1].astype(float)
    valid_rows   = ~np.isnan(numeric_data[:, :10]).any(axis=1)
    numeric_data = numeric_data[valid_rows]

    if numeric_data.shape[0] == 0:
        print(f"  [WARNING] No valid rows remain for {trial_type} after NaN removal.")
        return np.array([])

    mean    = np.nanmean(numeric_data[:, :10], axis=0)
    std_dev = np.nanstd(numeric_data[:, :10],  axis=0)
    mask    = np.all(np.abs(numeric_data[:, :10] - mean) <= 3 * std_dev, axis=1)

    viable_trials = np.count_nonzero(mask)
    filtered_data = numeric_data[mask]
    n_removed     = numeric_data.shape[0] - filtered_data.shape[0]

    if n_removed > 0:
        print(f"  [OUTLIER REMOVAL] {trial_type}: removed {n_removed} row(s) → {viable_trials} remain")

    is_normal        = all(
        len(numeric_data[:, i]) >= 3 and shapiro(numeric_data[:, i])[1] > 0.05
        for i in range(10)
    )
    normality_status = 'Normal' if is_normal else 'Not Normal'
    row_means        = np.round(np.nanmean(filtered_data, axis=0), 2)
    row_stds         = np.round(np.nanstd(filtered_data,  axis=0), 2)
    new_row          = [trial_type, viable_trials, percentile_outliers,
                        row_means.tolist(), row_stds.tolist(), normality_status]

    csv_path     = os.path.join(stats_dir, "metrics_stats_across_timepoints.csv")
    updated_rows = []
    found        = False
    try:
        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == trial_type:
                    updated_rows.append(new_row)
                    found = True
                else:
                    updated_rows.append(row)
    except FileNotFoundError:
        pass
    if not found:
        updated_rows.append(new_row)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

    return filtered_data


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2)


# =========================================
# STATISTICS
# =========================================

def run_statistics(extracted_values_dict, folders, stats_dir, side):
    all_results                = {}
    healthy_comparison_results = []
    kruskal_results            = {}   # ← NEW: store H and p per metric

    groups = [
        np.array([row for row in extracted_values_dict[folder]], dtype=float)
        for folder in folders
    ]
    groups = [g for g in groups if len(g) > 0]

    folder_labels = [
        folder.split('/')[-1].replace('_', ' ').replace("P0", "Healthy")
        for folder in folders
    ]

    if len(groups) == 2 and compare_to_healthy == 1:
        for i in range(groups[0].shape[1]):
            try:
                stat, p_val = wilcoxon(groups[0][:, i], groups[1][:, i])
            except Exception:
                p_val = np.nan
            healthy_comparison_results.append(p_val)

    elif len(groups) > 2:
        for i in range(groups[0].shape[1]):
            data_for_dunn = [group[:, i] for group in groups]

            # ── Kruskal-Wallis ──────────────────────────────────────────
            try:
                H, p_kw = kruskal(*data_for_dunn)
            except Exception:
                H, p_kw = np.nan, np.nan
            kruskal_results[METRIC_COLUMN_LABELS[i]] = {"H": H, "p": p_kw}
            # ────────────────────────────────────────────────────────────

            dunn_results  = sp.posthoc_dunn(data_for_dunn, p_adjust='bonferroni')
            dunn_results  = dunn_results.map(lambda x: f"{x:.2e}")
            dunn_results.index   = folder_labels
            dunn_results.columns = folder_labels
            all_results[METRIC_COLUMN_LABELS[i]] = dunn_results

        csv_path = os.path.join(stats_dir, f"statistical_comparison_{side}_{compare_trials}.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # ── Write KW results at top of CSV ──────────────────────────
            writer.writerow(["Kruskal-Wallis results"])
            writer.writerow(["Metric", "H-statistic", "p-value"])
            for metric_name, kw in kruskal_results.items():
                writer.writerow([metric_name, f"{kw['H']:.4f}", f"{kw['p']:.4e}"])
            writer.writerow([])
            # ────────────────────────────────────────────────────────────
            for test_name, df in all_results.items():
                writer.writerow([f"Dunn's test results for {test_name}"])
                writer.writerow([''] + list(df.columns))
                for index, row in df.iterrows():
                    writer.writerow([index] + row.tolist())
                writer.writerow([])

    return all_results, healthy_comparison_results, kruskal_results   # ← return kruskal too






def run_statistics_flx_ext(extracted_values_dict, folders, stats_dir, side, compare_trials):
    results      = [np.nan] * 5
    metric_pairs = [
        (0, 1, "Avg Max Neuron Firing Rate"),
        (2, 3, "Avg On-Cycle Neuron Firing Rate"),
        (4, 5, "Avg Off-Cycle Neuron Firing Rate"),
        (6, 7, "Freq"),
        (8, 9, "Burst Duration")
    ]
    rows = []
    for idx, (flx_idx, ext_idx, metric_name) in enumerate(metric_pairs):
        flx_vals, ext_vals = [], []
        for folder in folders:
            if folder not in extracted_values_dict:
                continue
            for row in extracted_values_dict[folder]:
                if np.isfinite(row[flx_idx]):
                    flx_vals.append(row[flx_idx])
                if np.isfinite(row[ext_idx]):
                    ext_vals.append(row[ext_idx])
        if len(flx_vals) < 2 or len(ext_vals) < 2:
            rows.append([metric_name, np.nan])
            continue
        try:
            _, p         = mannwhitneyu(flx_vals, ext_vals, alternative='two-sided')
            results[idx] = p
            rows.append([metric_name, p])
        except Exception:
            rows.append([metric_name, np.nan])

    try:
        os.makedirs(stats_dir, exist_ok=True)
        csv_path = os.path.join(stats_dir, f"statistical_comparison_flx_ext_{side}_{compare_trials}.csv")
        pd.DataFrame(rows, columns=["Metric", "p_value"]).to_csv(csv_path, index=False)
    except Exception:
        pass
    return results


def run_statistics_inter_phase(extracted_values_inter, folders, stats_dir, pop_type):
    groups = [
        np.array(extracted_values_inter.get(folder, []), dtype=float)
        for folder in folders
    ]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 3]

    folder_labels = [
        f.split('/')[-1].replace('_', ' ').replace("P0", "Healthy")
        for f in folders
    ]

    result = None
    kw_H   = np.nan
    kw_p   = np.nan

    if len(groups) == 2 and compare_to_healthy == 1:
        _, p   = wilcoxon(groups[0], groups[1])
        result = p
    elif len(groups) > 2:
        kw_H, kw_p = kruskal(*groups)
        dunn        = sp.posthoc_dunn(groups, p_adjust='bonferroni')
        dunn.index   = folder_labels[:len(groups)]
        dunn.columns = folder_labels[:len(groups)]
        result       = dunn

    csv_path = os.path.join(stats_dir, f"statistical_comparison_inter_phase_{pop_type}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Kruskal-Wallis (Inter-Phase)"])
        writer.writerow(["H-statistic", "p-value"])
        writer.writerow([
            f"{kw_H:.4f}" if not np.isnan(kw_H) else "NaN",
            f"{kw_p:.4e}" if not np.isnan(kw_p) else "NaN",
        ])
        writer.writerow([])
        if isinstance(result, pd.DataFrame):
            writer.writerow(["Inter-Phase Dunn posthoc (Bonferroni)"])
            writer.writerow([""] + list(result.columns))
            for idx, row in result.iterrows():
                writer.writerow([idx] + row.map(lambda x: f"{x:.2e}").tolist())
        elif result is not None:
            writer.writerow(["Wilcoxon p-value", f"{result:.2e}"])

    return result, kw_H, kw_p

# =========================================
# SIGNIFICANCE ANNOTATION HELPERS
# =========================================

def annotate_significance(ax, cv_data_flx, cv_data_ext, xtick_positions, metric_indices,
                          significance_results, significance_levels, pairs_to_compare,
                          metric_ymax, folders):
    if metric_indices is None or len(metric_indices) < 2:
        return
    if significance_results is None or len(folders) <= 2:
        return
    try:
        metric_flx = METRIC_COLUMN_LABELS[metric_indices[0]]
        metric_ext = METRIC_COLUMN_LABELS[metric_indices[1]]
    except Exception:
        return
    if metric_flx not in significance_results or metric_ext not in significance_results:
        return

    buffer       = 0.2 * metric_ymax
    global_y_max = metric_ymax + buffer
    ax.set_ylim([0, global_y_max + buffer])

    for (i, j) in pairs_to_compare:
        try:
            p_flx = float(significance_results[metric_flx].iloc[i, j])
            p_ext = float(significance_results[metric_ext].iloc[i, j])
        except Exception:
            continue

        sig_flx = next((s for t, s in significance_levels.items() if p_flx <= float(t)),
                       f"{p_flx:.3f}")
        if sig_flx:
            x1, x2 = xtick_positions[i * 2], xtick_positions[j * 2]
            ax.plot([x1, x1, x2, x2],
                    [global_y_max, global_y_max + 0.025, global_y_max + 0.025, global_y_max],
                    color='black')
            ax.vlines(x1, global_y_max - (global_y_max * .025), global_y_max, color='blue',   linewidth=2)
            ax.vlines(x2, global_y_max - (global_y_max * .025), global_y_max, color='blue',   linewidth=2)
            ax.text(x2, global_y_max + 0.1, sig_flx, ha='center', va='bottom', color='blue')

        sig_ext = next((s for t, s in significance_levels.items() if p_ext <= float(t)),
                       f"{p_ext:.3f}")
        if sig_ext:
            x1, x2 = xtick_positions[i * 2 + 1], xtick_positions[j * 2 + 1]
            ax.plot([x1, x1, x2, x2],
                    [global_y_max, global_y_max + 0.025, global_y_max + 0.025, global_y_max],
                    color='black')
            ax.vlines(x1, global_y_max - (global_y_max * .025), global_y_max, color='orange', linewidth=2)
            ax.vlines(x2, global_y_max - (global_y_max * .025), global_y_max, color='orange', linewidth=2)
            ax.text(x2, global_y_max + 0.1, sig_ext, ha='center', va='bottom', color='orange')


def annotate_inter_phase_significance(ax, cv_data, xtick_positions,
                                      significance_result, significance_levels,
                                      pairs_to_compare, metric_ymax):
    y_max  = 190
    buffer = 0.2 * y_max
    y_max += buffer
    ax.set_ylim([0, y_max + buffer])

    for (i, j) in pairs_to_compare:
        if isinstance(significance_result, float):
            p_value = significance_result
        elif isinstance(significance_result, pd.DataFrame):
            p_value = float(significance_result.iloc[i, j])
        else:
            continue
        sig = next((s for t, s in significance_levels.items() if p_value <= float(t)),
                   f"{p_value:.3f}")
        if sig:
            x1, x2 = xtick_positions[i], xtick_positions[j]
            ax.vlines(x1, y_max - (y_max * .05), y_max, color='black', linewidth=2)
            ax.vlines(x2, y_max - (y_max * .05), y_max, color='black', linewidth=2)
            ax.plot([x1, x1, x2, x2], [y_max, y_max + 0.05, y_max + 0.05, y_max], color='black')
            ax.text(x2, y_max + 0.05, sig, ha='center', va='bottom', color='black')


def annotate_significance_flx_ext(ax, cv_data_flx, cv_data_ext, xtick_positions,
                                  metric_indices, significance_results,
                                  significance_levels, metric_ymax):
    if significance_results is None:
        return
    if not isinstance(significance_results, (list, tuple, np.ndarray)):
        return
    if metric_indices is None or len(metric_indices) < 2:
        return

    metric_pair_to_result_index = {
        (0, 1): 0, (2, 3): 1, (4, 5): 2, (6, 7): 3, (8, 9): 4
    }
    metric_pair = tuple(metric_indices)
    if metric_pair not in metric_pair_to_result_index:
        return
    index = metric_pair_to_result_index[metric_pair]
    if index >= len(significance_results):
        return
    p_value = significance_results[index]
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return

    buffer       = 0.2 * metric_ymax
    global_y_max = metric_ymax + buffer
    ax.set_ylim([0, global_y_max + buffer])
    sig = next((s for t, s in significance_levels.items() if p_value <= float(t)),
               f"{p_value:.3f}")
    x1, x2 = xtick_positions[0], xtick_positions[1]
    ax.plot([x1, x1, x2, x2],
            [global_y_max, global_y_max + 0.05, global_y_max + 0.05, global_y_max],
            color='black')
    ax.text((x1 + x2) / 2, global_y_max + 0.08, sig,
            ha='center', va='bottom', color='black')


# =========================================
# FREQUENCY ZOOM HELPER
# Must be called AFTER all annotations since annotations call set_ylim internally.
# =========================================

def _apply_freq_zoom(ax, metric_title, metric_indices, extracted_values_dict, folders):
    """
    Zoom the frequency panel y-axis tightly around the actual data.
    Called as the very last step inside plot_metric so nothing overwrites it.
    """
    if metric_title != "Frequency":
        return
    freq_vals = []
    for folder in folders:
        for row in extracted_values_dict.get(folder, []):
            for col_idx in metric_indices:
                v = row[col_idx]
                if not np.isnan(v):
                    freq_vals.append(v)
    if not freq_vals:
        return
    lo = np.percentile(freq_vals, 5)
    hi = np.percentile(freq_vals, 95)
    margin = max((hi - lo) * 0.20, 0.05)   # 15% of spread, min 0.05 Hz
    ax.set_ylim(lo - margin, hi + margin)


def plot_metric(ax, title, ylabel, metric_ymax, metric_indices, folders, extracted_values_dict,
                scatter_color, group_spacing, pair_offset, significance_results,
                significance_levels, pairs_to_compare, annotation_type):

    cv_data_flx     = []
    cv_data_ext     = []
    xticks_combined = []

    for folder in folders:
        raw_flx = [
            t[metric_indices[0]] for t in extracted_values_dict[folder]
            if not np.isnan(t[metric_indices[0]])
        ]
        raw_ext = [
            t[metric_indices[1]] for t in extracted_values_dict[folder]
            if not np.isnan(t[metric_indices[1]])
        ]

        if title == "Frequency":
            print(f"{folder} RAW MAX FLX:", max(raw_flx) if raw_flx else None)
            print(f"{folder} RAW MAX EXT:", max(raw_ext) if raw_ext else None)

            MAX_FREQ = 3.5
            MIN_FREQ = 2.0
            raw_flx = [v for v in raw_flx if MIN_FREQ <= v < MAX_FREQ]
            raw_ext = [v for v in raw_ext if MIN_FREQ <= v < MAX_FREQ]
            folder_flx_data = _iqr_filter(raw_flx, k=1.5)
            folder_ext_data = _iqr_filter(raw_ext, k=1.5)
        else:
            folder_flx_data = raw_flx
            folder_ext_data = raw_ext

        cv_data_flx.append(folder_flx_data)
        cv_data_ext.append(folder_ext_data)

        test_type   = folder.split('/')[0]
        folder_name = folder.split('/')[-1].replace("_", " ")
        xtick_label = folder_name.replace("D1", "").replace("D2", "").strip()
        for substring, mapped_label in sorted(label_mapping.items(), key=lambda x: -len(x[0])):
            if substring in test_type:
                xtick_label += mapped_label
                break
        xticks_combined.append(xtick_label)

    xticks_combined = [label.replace("P0", "Healthy") for label in xticks_combined]

    cv_data         = []
    xtick_positions = []
    xtick_centers   = []
    x_pos           = 1
    for flx, ext in zip(cv_data_flx, cv_data_ext):
        cv_data.extend([flx, ext])
        xtick_positions.extend([x_pos - pair_offset, x_pos + pair_offset])
        xtick_centers.append(x_pos)
        x_pos += group_spacing

    # ── Boxplot ──────────────────────────────────────────────────────────
    ax.boxplot(
        cv_data,
        positions=np.array(xtick_positions),
        widths=pair_offset * 1.2,
        patch_artist=True,
        boxprops=dict(facecolor='gray'),
        medianprops=dict(color='black'),
    )
    for i, (data, pos) in enumerate(zip(cv_data, xtick_positions)):
        if len(data) == 0:
            continue
        color = scatter_color[i % 2]
        ax.scatter(
            np.ones_like(data) * pos,
            data,
            color=color,
            alpha=0.7,
            edgecolors='black',
            zorder=3,
        )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    
    flx_ext_labels = []
    for label in xticks_combined:
        flx_ext_labels.extend([f"{label} (Flx)", f"{label} (Ext)"])
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(flx_ext_labels, rotation=45, ha='right', fontsize=axis_label_fontsize)

    all_vals = [v for group in cv_data for v in group if not np.isnan(v)]
    ax.set_ylim(bottom=0, top=max(all_vals) * 1.2 if all_vals else 1)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(axis_line_thickness)

    # ── Significance annotations ──────────────────────────────────────────
    if (annotation_type == 'timepoint' and len(folders) > 2) or \
       (annotation_type == 'timepoint' and compare_to_healthy == 1):
        annotate_significance(ax, cv_data_flx, cv_data_ext, xtick_positions, metric_indices,
                              significance_results, significance_levels, pairs_to_compare,
                              metric_ymax, folders)
    elif annotation_type == 'flx_ext':
        annotate_significance_flx_ext(ax, cv_data_flx, cv_data_ext, xtick_positions,
                                      metric_indices, significance_results,
                                      significance_levels, metric_ymax)

    # ── Frequency zoom (must be last — annotations call set_ylim internally) ──
    if title == "Frequency":
        _apply_freq_zoom(ax, title, metric_indices, extracted_values_dict, folders)





def plot_inter_phase_metric(ax, title, ylabel, metric_ymax, folders,
                             extracted_values_inter, scatter_color, group_spacing,
                             inter_phase_significance, significance_levels,
                             pairs_to_compare, annotation_type, pop_type):
    cv_data         = []
    xtick_positions = []
    xtick_labels    = []
    x_pos           = 1

    schema    = POP_SCHEMAS.get(pop_type, {})
    pop_file1 = os.path.splitext(schema.get("files", ["pop1"])[0])[0].lstrip('_').upper()

    for folder in folders:
        folder_data = [v for v in extracted_values_inter.get(folder, []) if not np.isnan(v)]
        cv_data.append(folder_data)
        xtick_positions.append(x_pos)
        label = folder.split('/')[-1].replace("_", " ").replace("D1", "").replace("D2", "").strip()
        xtick_labels.append(label.replace("P0", "Healthy"))
        x_pos += group_spacing

    ax.boxplot(cv_data, positions=np.array(xtick_positions), patch_artist=True,
               boxprops=dict(facecolor='gray'), medianprops=dict(color='black'))
    for data, pos in zip(cv_data, xtick_positions):
        ax.scatter(np.ones(len(data)) * pos, data,
                   color=scatter_color[2], alpha=0.35, edgecolors='black', zorder=3)

    schema_files = schema.get("files", ["pop1", "pop2"])
    stem1        = os.path.splitext(schema_files[0])[0].lstrip('_').upper()
    ax.set_title(f"Interhemicord Phase", fontsize=24)
    ax.set_ylabel("Inter-Phase (°)", fontsize=17)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)

    all_vals = [v for group in cv_data for v in group if not np.isnan(v)]
    ax.set_ylim(bottom=0, top=max(all_vals) * 1.2 if all_vals else 1)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(axis_line_thickness)

    if inter_phase_significance is not None and (
        (annotation_type == 'timepoint' and len(folders) > 2) or
        (annotation_type == 'timepoint' and compare_to_healthy == 1)
    ):
        annotate_inter_phase_significance(
            ax, cv_data, xtick_positions,
            inter_phase_significance, significance_levels,
            pairs_to_compare, metric_ymax,
        )
    
def summarize_metrics(extracted_values_dict, folders, side, stats_dir):
    summary_rows = []
    for folder in folders:
        data  = np.array(extracted_values_dict[side][folder], dtype=float)
        label = folder.split("/")[-1].replace("_", " ").replace("P0", "Healthy")
        if data.size > 0:
            means = np.nanmean(data, axis=0)
            stds  = np.nanstd(data,  axis=0)
            row   = [label] + [f"{means[i]:.3f} ± {stds[i]:.3f}" for i in range(len(METRIC_COLUMN_LABELS))]
            summary_rows.append(row)

    csv_path = os.path.join(stats_dir, f"summary_stats_{side}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition"] + METRIC_COLUMN_LABELS)
        writer.writerows(summary_rows)
    print(f"  Saved summary stats → {csv_path}")


def plot_single_metric(folders, extracted_values_dict, side, metric_index, metric_name,
                       ylabel, ymax, output_dir, pop_type):
    fig, ax    = plt.subplots(figsize=(10, 6))
    data, labels = [], []
    for folder in folders:
        values = [row[metric_index] for row in extracted_values_dict[folder]
                  if not np.isnan(row[metric_index])]
        if not values:
            continue
        data.append(values)
        labels.append(folder.split("/")[-1].replace("_", " ").replace("P0", "Healthy"))

    ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightgray"))
    for i, vals in enumerate(data):
        ax.scatter(np.ones(len(vals)) * (i + 1), vals, color="black", alpha=0.45, zorder=3)
    ax.set_title(f"{side} – {metric_name}", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    all_vals = [v for group in data for v in group if not np.isnan(v)]
    ax.set_ylim(bottom=0, top=max(all_vals) * 1.2 if all_vals else 1)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{pop_type}_{side}_{metric_name.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_single_inter_phase(folders, extracted_values_inter, output_dir, pop_type,
                             inter_phase_significance, stats_dir):
    fig, ax             = plt.subplots(figsize=(10, 6))
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare    = [(0, i) for i in range(1, len(folders))]
    plot_inter_phase_metric(
        ax=ax, title="", ylabel="Inter-Phase (°)", metric_ymax=200, folders=folders,
        extracted_values_inter=extracted_values_inter,
        scatter_color=['blue', 'orange', 'black'], group_spacing=1.7,
        inter_phase_significance=inter_phase_significance,
        significance_levels=significance_levels,
        pairs_to_compare=pairs_to_compare, annotation_type='timepoint', pop_type=pop_type,
    )
    plt.tight_layout()

    out_path_png = os.path.join(output_dir, f"{pop_type}_inter_phase.png")
    out_path_svg = os.path.join(output_dir, f"{pop_type}_inter_phase.svg")
    
    if save_as_svg == 1: 
        
        plt.savefig(out_path_svg, format='svg', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved Collective Figure 1 {out_path_svg} as SVG")
    
    elif save_as_svg == 0: 

        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Collective Figure 1 {out_path_png} as PNG")
        
    rows = []
    for folder in folders:
        label  = folder.split("/")[-1].replace("P0", "Healthy")
        values = extracted_values_inter.get(folder, [])
        for i, v in enumerate(values, start=1):
            rows.append([label, i, v])
    csv_path = os.path.join(stats_dir, f"inter_phase_values_{pop_type}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Trial", "Inter_Phase_degrees"])
        writer.writerows(rows)
    print(f"  Saved inter-phase values → {csv_path}")


def sanity_check_single_trace(file_paths, pop_type):
    schema          = POP_SCHEMAS[pop_type]
    threshold       = schema["burst_threshold_raw"]
    time_resolution = schema["time_resolution"]
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None).to_numpy().flatten()
    time  = np.arange(len(data)) * time_resolution
    above = data > threshold
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]
    burst_regions, burst_midpoints = [], []
    if len(crossings) >= 2:
        if not above[crossings[0] + 1]:
            crossings = crossings[1:]
        for i in range(0, len(crossings) - 1, 2):
            start, end = crossings[i], crossings[i + 1]
            burst_regions.append((start, end))
            burst_midpoints.append((start + end) / 2)
    plt.figure(figsize=(14, 6))
    plt.plot(time, data, linewidth=1.25)
    plt.axhline(threshold, linestyle="--", color="red", linewidth=2, label="Threshold")
    for i, (start, end) in enumerate(burst_regions):
        plt.axvspan(start * time_resolution, end * time_resolution, alpha=0.25, color="green",
                    label="Burst Region" if i == 0 else "")
    for i, mid in enumerate(burst_midpoints):
        plt.scatter(mid * time_resolution, threshold, color="orange", s=50,
                    label="Burst Midpoint" if i == 0 else "")
    plt.title(f"Sanity Check — {pop_type}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# =========================================
# EXAMPLE TRACE HELPERS
# =========================================

def _load_first_trial_trace(folder_containing_data, folder, pop_type, side):
    import glob
    schema    = POP_SCHEMAS[pop_type]
    pop_files = schema["files"]
    patterns  = [os.path.join(folder_containing_data, folder, '**', f'{side}{f}')
                 for f in pop_files]
    matched   = [sorted(glob.glob(p, recursive=True)) for p in patterns]
    if not all(matched):
        return None, None, schema["num_neurons"]
    try:
        d1  = np.loadtxt(matched[0][0], delimiter=',', dtype=float)
        d2  = np.loadtxt(matched[1][0], delimiter=',', dtype=float)
        d1n = (d1 - d1.min()) / (d1.max() - d1.min() + 1e-12)
        d2n = (d2 - d2.min()) / (d2.max() - d2.min() + 1e-12)
        return d1n, d2n, schema["num_neurons"]
    except Exception as e:
        print(f"  [TRACE LOAD ERROR] {folder} {side}: {e}")
        return None, None, schema["num_neurons"]


def _plot_example_trace(ax, d1, d2, title, schema, pop_type, side):
    t_s   = np.arange(len(d1)) * schema["sampling_dt"] / 1000.0
    stem1 = os.path.splitext(schema.get("files", ["pop1", "pop2"])[0])[0].lstrip('_').upper()
    stem2 = os.path.splitext(schema.get("files", ["pop1", "pop2"])[1])[0].lstrip('_').upper()
    ax.plot(t_s, d1, color='royalblue',  linewidth=1.0, label=stem1, alpha=0.85)
    ax.plot(t_s, d2, color='darkorange', linewidth=1.0, label=stem2, alpha=0.85)
    ax.axhline(schema["burst_threshold"], color='red', linewidth=0.8,
               linestyle='--', label='Threshold', alpha=0.7)
    ax.set_title(title, fontsize=9, fontweight='normal')
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Norm. activity", fontsize=7)
    ax.tick_params(axis='both', labelsize=6, length=2)
    ax.legend(fontsize=5, loc='upper right', framealpha=0.5)
    ax.set_ylim(-0.05, 1.15)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


# =========================================
# OUTPUT DIRECTORY HELPERS
# =========================================

def get_side_output_dirs(base_dir, pop_type, side):
    base  = os.path.join(base_dir, "MULTI-SEED-ANALYSIS", pop_type, side)
    plots = os.path.join(base, "plots")
    stats = os.path.join(base, "stats")
    os.makedirs(plots, exist_ok=True)
    os.makedirs(stats, exist_ok=True)
    return plots, stats


def get_inter_phase_output_dirs(base_dir, pop_type):
    base  = os.path.join(base_dir, "MULTI-SEED-ANALYSIS", pop_type, "INTER_PHASE")
    plots = os.path.join(base, "plots")
    stats = os.path.join(base, "stats")
    os.makedirs(plots, exist_ok=True)
    os.makedirs(stats, exist_ok=True)
    return plots, stats


# =========================================
# COMBINED DISPERSION PLOT
# =========================================

def plot_comparison_cv_with_dispersion(folders, extracted_values_dict, significance_results,
                                       output_dir, side, pop_type,
                                       stats_dir, compare_trials,
                                       extracted_values_inter=None,
                                       inter_phase_significance=None):
    if output_dir is None:
        raise RuntimeError("output_dir must be provided")

    scatter_color       = ['blue', 'orange', 'black']
    group_spacing       = 1.7
    pair_offset         = 0.3
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare    = [(0, i) for i in range(1, len(folders))]

    metrics = [
        {"title": "Avg Max Neuron Firing Rate",       "ylabel": "Neuron Firing Rate", "metric_ymax": 150, "metric_indices": [0, 1]},
        {"title": "Avg On-Cycle Neuron Firing Rate",  "ylabel": "Neuron Firing Rate", "metric_ymax": 150, "metric_indices": [2, 3]},
        {"title": "Avg Off-Cycle Neuron Firing Rate", "ylabel": "Neuron Firing Rate", "metric_ymax": 150, "metric_indices": [4, 5]},
        {"title": "Frequency",                        "ylabel": "Freq (Hz)",          "metric_ymax": 3.5, "metric_indices": [6, 7]},
        {"title": "Burst Duration",                   "ylabel": "Time (ms)",          "metric_ymax": 350, "metric_indices": [8, 9]},
    ]

    # ---- Figure 1: significance across timepoints ----
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    axes      = axes.flatten()
    for i, metric in enumerate(metrics):
        plot_metric(
            ax=axes[i], title=metric["title"], ylabel=metric["ylabel"],
            metric_ymax=metric["metric_ymax"], metric_indices=metric["metric_indices"],
            folders=folders, extracted_values_dict=extracted_values_dict,
            scatter_color=scatter_color, group_spacing=group_spacing, pair_offset=pair_offset,
            significance_results=significance_results, significance_levels=significance_levels,
            pairs_to_compare=pairs_to_compare, annotation_type='timepoint',
        )
    if extracted_values_inter is not None:
        plot_inter_phase_metric(
            ax=axes[5], title="", ylabel="", metric_ymax=200, folders=folders,
            extracted_values_inter=extracted_values_inter, scatter_color=scatter_color,
            group_spacing=group_spacing, inter_phase_significance=inter_phase_significance,
            significance_levels=significance_levels, pairs_to_compare=pairs_to_compare,
            annotation_type='timepoint', pop_type=pop_type,
        )
    else:
        axes[5].set_visible(False)
    out_path = os.path.join(output_dir, f"{side}_combined_metrics_1_{pop_type}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # ---- Figure 2: Flx vs Ext significance ----
    significance_results_flx_ext = run_statistics_flx_ext(
        extracted_values_dict, folders, stats_dir, side, compare_trials
    )
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    axes      = axes.flatten()
    for i, metric in enumerate(metrics):
        plot_metric(
            ax=axes[i], title=metric["title"], ylabel=metric["ylabel"],
            metric_ymax=metric["metric_ymax"], metric_indices=metric["metric_indices"],
            folders=folders, extracted_values_dict=extracted_values_dict,
            scatter_color=scatter_color, group_spacing=group_spacing, pair_offset=pair_offset,
            significance_results=significance_results_flx_ext, significance_levels=significance_levels,
            pairs_to_compare=pairs_to_compare, annotation_type='flx_ext',
        )
    if extracted_values_inter is not None:
        plot_inter_phase_metric(
            ax=axes[5], title="", ylabel="", metric_ymax=200, folders=folders,
            extracted_values_inter=extracted_values_inter, scatter_color=scatter_color,
            group_spacing=group_spacing, inter_phase_significance=inter_phase_significance,
            significance_levels=significance_levels, pairs_to_compare=pairs_to_compare,
            annotation_type='timepoint', pop_type=pop_type,
        )
    else:
        axes[5].set_visible(False)
    out_path = os.path.join(output_dir, f"{side}_combined_metrics_2_{pop_type}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================
# COLLECTIVE FIGURES
# =========================================

def _reformat_ax_compact(ax, title, ylabel, title_fs=24, label_fs=20, tick_fs=18, pad=4):
    ax.set_title(title, fontsize=title_fs, fontweight='normal', pad=pad)
    ax.set_ylabel(ylabel, fontsize=label_fs, fontweight='normal')
    ax.set_xlabel("")
    ax.tick_params(axis='both', labelsize=tick_fs, length=3, pad=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fs)
    for txt in ax.texts:
        txt.set_fontsize(tick_fs + 1)
    for sp in ax.spines.values():
        sp.set_linewidth(2)    
    
def _get_freq_range(extracted_values_dict, folders, metric_indices, margin_frac=0.15):
    """
    Returns (ymin, ymax) for frequency panels based on actual data,
    with a percentage margin added above and below.
    Filters out values >= 50 Hz (same logic as plot_metric).
    """
    all_vals = []
    for side in ["LEFT", "RIGHT"]:
        for folder in folders:
            for row in extracted_values_dict[side].get(folder, []):
                for col_idx in metric_indices:
                    v = row[col_idx]
                    if not np.isnan(v) and v < 50:
                        all_vals.append(v)
    if not all_vals:
        return 0, 5
    lo = np.percentile(all_vals, 5)
    hi = np.percentile(all_vals, 95)
    margin = max((hi - lo) * margin_frac, 0.1)
    return max(0, lo - margin), hi + margin

def plot_collective_figure_1(extracted_values_dict, folders, significance_results_per_side,
                              folder_containing_data, output_dir, pop_type, stats_dir,
                              extracted_values_inter=None, inter_phase_significance=None):
    scatter_color       = ['blue', 'orange', 'black']
    group_spacing       = 0.9
    pair_offset         = 0.2
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare    = [(0, i) for i in range(1, len(folders))]

    freq_ymin, freq_ymax = _get_freq_range(
        extracted_values_dict, folders, metric_indices=[6, 7], margin_frac=0.20
    )

    fig = plt.figure(figsize=(24, 18))
    gs  = GridSpec(
        3, 2,
        height_ratios=[1, 1, 1],
        hspace=0.65,
        wspace=0.30,
        top=0.91,
        bottom=0.10,
        left=0.08,
        right=0.98,
    )
    # ── Column headers ──────────────────────────────────────────────────────
    # x positions are the horizontal midpoints of the two columns in figure coords.
    # With left=0.11, right=0.97, wspace=0.32 and two equal columns the midpoints
    # sit at roughly 0.305 and 0.735.
    for x_mid, label in [(0.305, "LEFT HEMICORD"), (0.735, "RIGHT HEMICORD")]:
        fig.text(
            x_mid, 0.915,           # just above gs top=0.88
            label,
            ha='center', va='bottom',
            fontsize=16, fontweight='bold',
            color='black',
        )

    per_side_metrics = [
        (0, "Avg Max Firing Rate", "Firing rate",   150, [0, 1]),
        (1, "Frequency",           "Freq (Hz)",     3.5, [6, 7]),
        (2, "Burst Duration",      "Duration (ms)", 350, [8, 9]),
    ]

    for row, metric_name, ylabel, metric_ymax, metric_indices in per_side_metrics:
        for col, side in enumerate(["LEFT", "RIGHT"]):
            ax               = fig.add_subplot(gs[row, col])
            analyzed_folders = list(extracted_values_dict[side].keys())
            sig_res          = significance_results_per_side.get(side, {})

            plot_metric(
                ax=ax, title=metric_name, ylabel=ylabel,
                metric_ymax=metric_ymax, metric_indices=metric_indices,
                folders=analyzed_folders,
                extracted_values_dict=extracted_values_dict[side],
                scatter_color=scatter_color,
                group_spacing=group_spacing, pair_offset=pair_offset,
                significance_results=sig_res, significance_levels=significance_levels,
                pairs_to_compare=pairs_to_compare, annotation_type='timepoint',
            )

            if metric_name == "Frequency":
                ax.set_ylim(0, 5)
                ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

            if metric_name == "Avg Max Firing Rate":
                ax.set_ylim(0, 150)

    
    out_path_png = os.path.join(output_dir, f"collective_figure_1_{pop_type}.png")
    out_path_svg = os.path.join(output_dir, f"collective_figure_1_{pop_type}.svg")

    if save_as_svg == 1:
        plt.savefig(out_path_svg, format='svg', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved Collective Figure 1 {out_path_svg} as SVG")
    else:
        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Collective Figure 1 {out_path_png} as PNG")


def plot_collective_figure_2(extracted_values_dict, folders, significance_results_per_side,
                              extracted_values_inter, inter_phase_significance,
                              output_dir, pop_type, stats_dir):
    """
    Inter-phase boxplot + colour-coded p-value table.
    """
    scatter_color       = ['blue', 'orange', 'black']
    group_spacing       = 0.9
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare    = [(0, i) for i in range(1, len(folders))]

    schema = POP_SCHEMAS.get(pop_type, {})
    stem1  = os.path.splitext(schema.get("files", ["pop1"])[0])[0].lstrip('_').upper()

    folder_labels = [f.split('/')[-1].replace('_', ' ').replace("P0", "Healthy") for f in folders]

    pval_rows = []
    if isinstance(inter_phase_significance, float):
        p     = inter_phase_significance
        stars = next((s for t, s in significance_levels.items() if p <= float(t)), "ns")
        pval_rows.append((folder_labels[0], folder_labels[1], p, stars))
    elif isinstance(inter_phase_significance, pd.DataFrame):
        df = inter_phase_significance
        n  = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    p = float(df.iloc[i, j])
                except Exception:
                    p = np.nan
                stars = next((s for t, s in significance_levels.items()
                              if not np.isnan(p) and p <= float(t)), "ns")
                pval_rows.append((folder_labels[i], folder_labels[j], p, stars))

    n_pval_rows = max(len(pval_rows), 1)
    fig = plt.figure(figsize=(12, 7 + n_pval_rows * 0.35))

    if pval_rows:
        gs = GridSpec(2, 1, height_ratios=[3, max(1, n_pval_rows * 0.55)],
                      hspace=0.55, top=0.90, bottom=0.05, left=0.10, right=0.97)
        ax       = fig.add_subplot(gs[0, 0])
        ax_table = fig.add_subplot(gs[1, 0])
    else:
        gs = GridSpec(1, 1, top=0.88, bottom=0.20, left=0.10, right=0.97)
        ax       = fig.add_subplot(gs[0, 0])
        ax_table = None

    plot_inter_phase_metric(
        ax=ax, title="", ylabel="Phase Difference (°)", metric_ymax=200, folders=folders,
        extracted_values_inter=extracted_values_inter, scatter_color=scatter_color,
        group_spacing=group_spacing, inter_phase_significance=inter_phase_significance,
        significance_levels=significance_levels, pairs_to_compare=pairs_to_compare,
        annotation_type='timepoint', pop_type=pop_type,
    )


    _reformat_ax_compact(ax, f"Inter-Phase — Left {stem1} vs Right {stem1}", "Phase (°)",
                         title_fs=10, label_fs=9, tick_fs=8)

    if ax_table is not None and pval_rows:
        ax_table.axis('off')
        col_headers = ["Comparison", "p-value", "Significance"]
        table_data  = [
            [f"{r[0]}  vs  {r[1]}", f"{r[2]:.4f}" if not np.isnan(r[2]) else "NaN", r[3]]
            for r in pval_rows
        ]
        row_colours = [
            ['#d4edda'] * 3 if r[3] in ('***', '**', '*') else ['#f2f2f2'] * 3
            for r in pval_rows
        ]
        tbl = ax_table.table(
            cellText=table_data, colLabels=col_headers,
            cellColours=row_colours, colColours=['#c8c8c8'] * 3,
            cellLoc='center', loc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        for col_idx in range(len(col_headers)):
            tbl[(0, col_idx)].set_text_props(fontweight='bold')
        ax_table.set_title("Inter-Phase Statistical Comparisons",
                           fontsize=9, fontweight='bold', pad=6)

    out_path_png = os.path.join(output_dir, f"collective_figure_2_{pop_type}.png")
    out_path_svg = os.path.join(output_dir, f"collective_figure_2_{pop_type}.svg")

    if save_as_svg == 1:
        plt.savefig(out_path_svg, format='svg', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved Collective Figure 2 {out_path_svg} as SVG")

    elif save_as_svg == 0:
        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Collective Figure 2 {out_path_png} as PNG")
    
def plot_collective_summary(extracted_values_dict, folders, significance_results_per_side,
                            extracted_values_inter, inter_phase_significance,
                            output_dir, pop_type, stats_dir):
    scatter_color       = ['blue', 'orange', 'black']
    group_spacing       = 0.9
    pair_offset         = 0.2
    significance_levels = {'0.001': '***', '0.01': '**', '0.05': '*'}
    pairs_to_compare    = [(0, i) for i in range(1, len(folders))]

    fig = plt.figure(figsize=(8.27, 11.69))
    gs  = fig.add_gridspec(
        5, 2, height_ratios=[1, 1, 1, 1, 1.1],
        hspace=0.70, wspace=0.38,
        top=0.93, bottom=0.05, left=0.11, right=0.97,
    )

    per_side_metrics = [
        (0, "Avg Max Firing Rate", "Firing rate",   150, [0, 1]),
        (1, "Frequency",           "Freq (Hz)",     3.5, [6, 7]),
        (2, "Burst Duration",      "Duration (ms)", 350, [8, 9]),
    ]

    for row, metric_name, ylabel, metric_ymax, metric_indices in per_side_metrics:
        for col, side in enumerate(["LEFT", "RIGHT"]):
            ax               = fig.add_subplot(gs[row, col])
            analyzed_folders = list(extracted_values_dict[side].keys())
            sig_res          = significance_results_per_side.get(side, {})
            if len(metric_indices) == 1:
                ax.set_visible(False)
                continue
            plot_metric(
                ax=ax, title=metric_name, ylabel=ylabel,
                metric_ymax=metric_ymax, metric_indices=metric_indices,
                folders=analyzed_folders,
                extracted_values_dict=extracted_values_dict[side],
                scatter_color=scatter_color,
                group_spacing=group_spacing, pair_offset=pair_offset,
                significance_results=sig_res, significance_levels=significance_levels,
                pairs_to_compare=pairs_to_compare, annotation_type='timepoint',
            )

            # Override frequency y-axis LAST so nothing else stomps on it
            # if metric_name == "Frequency":
            #     freq_ymin = 0.5
            #     freq_ymax = 4
            #     ax.set_ylim(freq_ymin, freq_ymax)
            #     ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

            _reformat_ax_compact(ax, metric_name, ylabel)

    fig.text(0.305, 0.955, "LEFT HEMICORD",
            ha='center', va='bottom', fontsize=13, fontweight='bold')

    fig.text(0.735, 0.955, "RIGHT HEMICORD",
            ha='center', va='bottom', fontsize=13, fontweight='bold')

    fig.add_artist(plt.Line2D([0.535, 0.535], [0.10, 0.82],
                               transform=fig.transFigure, color='lightgray',
                               linewidth=0.6, linestyle='--'))

    out_path_png = os.path.join(output_dir, f"collective_summary_{pop_type}.png")
    out_path_svg = os.path.join(output_dir, f"collective_summmary_{pop_type}.svg")

    if save_as_svg == 1:
        plt.savefig(out_path_svg, format='svg', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved Collective Summary {out_path_svg} as SVG")

    elif save_as_svg == 0:
        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Collective Summary {out_path_png} as PNG")


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    folder_containing_data = sys.argv[1]
    pop_type               = sys.argv[2]
    folders                = sys.argv[3:]

    if pop_type not in POP_SCHEMAS:
        raise ValueError(f"Unknown pop_type '{pop_type}'. Options: {list(POP_SCHEMAS.keys())}")

    schema          = POP_SCHEMAS[pop_type]
    analysis_config = {
        "burst_threshold":   schema["burst_threshold"],
        "min_peak_distance": schema["min_peak_distance"],
        "num_motor_neurons": schema["num_neurons"],
    }

    # =========================================
    # STEP 1 — DATA EXTRACTION
    # =========================================
    extracted_values_dict = {}
    _section("STEP 1 — DATA EXTRACTION")

    for side in ["LEFT", "RIGHT"]:
        extracted_values_dict[side] = {}
        _, stats_dir = get_side_output_dirs(folder_containing_data, pop_type, side)

        for folder in folders:
            data_array  = []
            file_groups = find_population_files(folder_containing_data, folder, pop_type, side)

            print(f"[DEBUG] {side} {folder}: "
            f"{len(file_groups)} trials found")

            for file_list, trial_name in file_groups:
                try:
                    if schema["type"] == "paired":
                        data_row = analyze_output(
                            file_list[0], file_list[1], pop_type,
                            y_line_bd=analysis_config["burst_threshold"],
                            y_line_phase=analysis_config["burst_threshold"],
                            min_dist=analysis_config["min_peak_distance"],
                            num_motor_neurons=analysis_config["num_motor_neurons"],
                        )
                    else:
                        data_row = analyze_individual_population(
                            file_list[0], pop_type,
                            y_line_bd=analysis_config["burst_threshold"],
                            min_dist=analysis_config["min_peak_distance"],
                            num_neurons=analysis_config["num_motor_neurons"],
                        )
                    if sanitycheck_parameters:
                        sanity_check_single_trace(file_list, pop_type)
                        break
                except Exception as e:
                    print(f"  [ERROR] Trial {trial_name}: {e}")
                    data_row = [np.nan] * 11

                data_row = list(np.round(data_row, 4))
                data_row.append(trial_name)
                data_array.append(data_row)

            trial_type = f"{folder.split('/')[-1]}_{side}"
            data_array = remove_outliers(trial_type, data_array, stats_dir)
            extracted_values_dict[side][folder] = data_array

    # Compact extraction summary
    _section("STEP 1 — EXTRACTION SUMMARY (mean ± SD per condition)")
    for side in ["LEFT", "RIGHT"]:
        _subsection(side)
        for folder in folders:
            label = folder.split('/')[-1].replace('P0', 'Healthy')
            _print_summary_table(label, extracted_values_dict[side][folder])

    # =========================================
    # STEP 1b — INTER-PHASE EXTRACTION
    # =========================================
    _section("STEP 1b — INTER-PHASE EXTRACTION (LEFT MNP1 vs RIGHT MNP1)")
    extracted_values_inter = {}
    inter_plots_dir, inter_stats_dir = get_inter_phase_output_dirs(folder_containing_data, pop_type)

    for folder in folders:
        pairs      = find_inter_phase_file_pairs(folder_containing_data, folder, pop_type)
        inter_vals = []
        for left_f, right_f, trial_name in pairs:
            ip = compute_inter_phase(
                left_f, right_f, pop_type,
                y_line_phase=analysis_config["burst_threshold"],
                min_dist=analysis_config["min_peak_distance"],
            )
            inter_vals.append(ip)

        arr = np.array(inter_vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 1:
            mean, sd = np.nanmean(arr), np.nanstd(arr)
            mask     = np.abs(arr - mean) <= 3 * sd
            n_rm     = np.sum(~mask)
            if n_rm > 0:
                print(f"  [OUTLIER REMOVAL inter-phase] {folder}: removed {n_rm} value(s)")
            arr = arr[mask]
        extracted_values_inter[folder] = arr.tolist()

    # Inter-phase summary table
    print(f"\n  {'Condition':<20}  {'n':>4}  {'Mean (°)':>10}  {'SD (°)':>10}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*10}  {'-'*10}")
    for folder in folders:
        label = folder.split('/')[-1].replace('P0', 'Healthy')
        vals  = [v for v in extracted_values_inter[folder] if not np.isnan(v)]
        if vals:
            print(f"  {label:<20}  {len(vals):>4}  {np.mean(vals):>10.3f}  {np.std(vals):>10.3f}")
        else:
            print(f"  {label:<20}  {'0':>4}  {'NaN':>10}  {'NaN':>10}")


    
    # ── Intra-phase extraction (flx vs ext per hemicord) ──────────────────
    _section("STEP 1b — INTRA-PHASE EXTRACTION (Flx vs Ext per hemicord)")
    intra_phase_dict = {"LEFT": {}, "RIGHT": {}}
 
    for side in ["LEFT", "RIGHT"]:
        for folder in folders:
            file_groups = find_population_files(
                folder_containing_data, folder, pop_type, side
            )
            all_cycle_phases = []
            for file_list, trial_name in file_groups:
                if len(file_list) < 2:
                    continue
                flx_file = file_list[0]   # _output_mnp1.csv  (flexor)
                ext_file = file_list[1]   # _output_mnp2.csv  (extensor)
                cycle_phases, mean_deg = compute_intra_phase_per_cycle(
                    flx_file, ext_file, pop_type
                )
                all_cycle_phases.extend(cycle_phases)
 
            intra_phase_dict[side][folder] = all_cycle_phases
 
            tp_label = folder.split('/')[-1].replace('P0', 'Healthy')
            n        = len(all_cycle_phases)
            if n > 0:
                print(
                    f"  {side:<6} {tp_label:<15}  n={n:>4} cycles  "
                    f"mean={np.nanmean(all_cycle_phases):.1f}°  "
                    f"SD={np.nanstd(all_cycle_phases):.1f}°"
                )
            else:
                print(f"  {side:<6} {tp_label:<15}  no cycles detected")

    # =========================================
    # STEP 2 — STATISTICS + PLOTS
    # =========================================
    compare_trials                = "_vs_".join([f.split('/')[-1] for f in folders])
    significance_results_per_side = {}

    _section("STEP 2 — STATISTICS")

    inter_phase_significance, inter_phase_H, inter_phase_p_kw = run_statistics_inter_phase(
        extracted_values_inter, folders, inter_stats_dir, pop_type
    )

    # Inter-phase p-value table
    sig_map       = {'0.001': '***', '0.01': '**', '0.05': '*'}
    folder_labels = [f.split('/')[-1].replace('P0', 'Healthy') for f in folders]
    
    _subsection("Inter-phase p-values")
    # Print KW H if available (>2 groups)
    if not np.isnan(inter_phase_H):
        print(f"  Kruskal-Wallis:  H = {inter_phase_H:.4f},  p = {inter_phase_p_kw:.4e}")
        print()
    print(f"  {'Comparison':<40}  {'p-value':>10}  {'Sig':>5}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*5}")
    
    if isinstance(inter_phase_significance, float):
        p     = inter_phase_significance
        stars = next((s for t, s in sig_map.items() if p <= float(t)), "ns")
        print(f"  {folder_labels[0]:<18} vs {folder_labels[1]:<18}  {p:>10.4f}  {stars:>5}")
    elif isinstance(inter_phase_significance, pd.DataFrame):
        df = inter_phase_significance
        n  = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    p = float(df.iloc[i, j])
                except Exception:
                    p = np.nan
                stars = next((s for t, s in sig_map.items()
                              if not np.isnan(p) and p <= float(t)), "ns")
                p_str = f"{p:.4f}" if not np.isnan(p) else "NaN"
                comp  = f"{folder_labels[i]:<18} vs {folder_labels[j]:<18}"
                print(f"  {comp}  {p_str:>10}  {stars:>5}")

    for side in ["LEFT", "RIGHT"]:
        analyzed_folders = list(extracted_values_dict[side].keys())
        if not analyzed_folders:
            continue

        plots_dir, stats_dir = get_side_output_dirs(folder_containing_data, pop_type, side)

        significance_results, _, kruskal_results = run_statistics(   # ← unpack 3 values now
            extracted_values_dict[side], analyzed_folders, stats_dir=stats_dir, side=side
        )
        significance_results_per_side[side] = significance_results

        wilcoxon_flx_ext = run_statistics_wilcoxon_flx_ext(
            extracted_values_dict[side], analyzed_folders, stats_dir, side, compare_trials
        )

        metric_label_pairs = [
            ("Avg Max Neuron Firing Rate Flx", "Max FR Flx"),
            ("Avg Max Neuron Firing Rate Ext", "Max FR Ext"),
            ("Freq Flx",                       "Freq Flx"),
            ("Freq Ext",                       "Freq Ext"),
            ("Burst Duration Flx",             "Burst Dur Flx"),
            ("Burst Duration Ext",             "Burst Dur Ext"),
        ]

        # KW labels need to match METRIC_COLUMN_LABELS keys exactly:
        kw_label_pairs = [
            ("Avg Max Neuron Firing Rate Flx", "Max FR Flx"),
            ("Avg Max Neuron Firing Rate Ext", "Max FR Ext"),
            ("Freq Flx",                       "Freq Flx"),
            ("Freq Ext",                       "Freq Ext"),
            ("Burst Duration Flx",             "Burst Dur Flx"),
            ("Burst Duration Ext",             "Burst Dur Ext"),
        ]

        _print_pvalue_table(
            side, analyzed_folders, significance_results, metric_label_pairs,
            kruskal_results=kruskal_results,
            wilcoxon_flx_ext=wilcoxon_flx_ext,
        )

        plot_comparison_cv_with_dispersion(
            analyzed_folders, extracted_values_dict[side], significance_results,
            output_dir=plots_dir, side=side, pop_type=pop_type,
            stats_dir=stats_dir, compare_trials=compare_trials,
            extracted_values_inter=extracted_values_inter if side == "LEFT" else None,
            inter_phase_significance=inter_phase_significance if side == "LEFT" else None,
        )

        summarize_metrics(extracted_values_dict, analyzed_folders, side, stats_dir)

    # =========================================
    # STEP 3 — SINGLE-METRIC PLOTS
    # =========================================
    _section("STEP 3 — SINGLE-METRIC PLOTS")
    for side in ["LEFT", "RIGHT"]:
        analyzed_folders = list(extracted_values_dict[side].keys())
        if not analyzed_folders:
            continue
        plots_dir, _ = get_side_output_dirs(folder_containing_data, pop_type, side)
        single_metrics = [
            (0, "Avg Max Neuron Firing Rate (Flexor)",   "Rate",      150),
            (1, "Avg Max Neuron Firing Rate (Extensor)",  "Rate",      150),
            (2, "Avg On-Cycle Firing Rate (Flexor)",      "Rate",      150),
            (3, "Avg On-Cycle Firing Rate (Extensor)",    "Rate",      150),
            (4, "Avg Off-Cycle Firing Rate (Flexor)",     "Rate",      150),
            (5, "Avg Off-Cycle Firing Rate (Extensor)",   "Rate",      150),
            (6, "Frequency (Flexor)",                     "Hz",        3.5),
            (7, "Frequency (Extensor)",                   "Hz",        3.5),
            (8, "Burst Duration (Flexor)",                "Time (ms)", 350),
            (9, "Burst Duration (Extensor)",              "Time (ms)", 350),
        ]
        for idx, name, ylabel, ymax in single_metrics:
            plot_single_metric(
                folders=analyzed_folders,
                extracted_values_dict=extracted_values_dict[side],
                side=side, metric_index=idx, metric_name=name,
                ylabel=ylabel, ymax=ymax,
                output_dir=plots_dir, pop_type=pop_type,
            )

    plot_single_inter_phase(
        folders=folders, extracted_values_inter=extracted_values_inter,
        output_dir=inter_plots_dir, pop_type=pop_type,
        inter_phase_significance=inter_phase_significance,
        stats_dir=inter_stats_dir,
    )

    # =========================================
    # STEP 4 — COLLECTIVE FIGURES
    # =========================================
    _section("STEP 4 — COLLECTIVE FIGURES")
    plots_dir_left, stats_dir_left = get_side_output_dirs(folder_containing_data, pop_type, "LEFT")

    plot_collective_figure_1(
        extracted_values_dict=extracted_values_dict, folders=folders,
        significance_results_per_side=significance_results_per_side,
        folder_containing_data=folder_containing_data,
        output_dir=plots_dir_left, pop_type=pop_type, stats_dir=stats_dir_left,
        extracted_values_inter=extracted_values_inter,
        inter_phase_significance=inter_phase_significance,
    )

    plot_collective_figure_2(
        extracted_values_dict=extracted_values_dict, folders=folders,
        significance_results_per_side=significance_results_per_side,
        extracted_values_inter=extracted_values_inter,
        inter_phase_significance=inter_phase_significance,
        output_dir=inter_plots_dir, pop_type=pop_type, stats_dir=inter_stats_dir,
    )

    plot_collective_summary(
        extracted_values_dict=extracted_values_dict, folders=folders,
        significance_results_per_side=significance_results_per_side,
        extracted_values_inter=extracted_values_inter,
        inter_phase_significance=inter_phase_significance,
        output_dir=plots_dir_left, pop_type=pop_type, stats_dir=stats_dir_left,
    )

    plot_intra_phase_hemicord(
        intra_phase_dict=intra_phase_dict,
        folders=folders,
        output_dir=inter_plots_dir,
        pop_type=pop_type,
        stats_dir=inter_stats_dir,
    )

    _section("ALL DONE")
    print(f"  Output root: {os.path.join(folder_containing_data, 'MULTI-SEED-ANALYSIS', pop_type)}")