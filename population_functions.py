#!/usr/bin/env python

#include <static_connection.h>
import time
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as pyplot
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.signal import convolve2d, windows, butter, filtfilt, decimate
import time
import copy
import set_network_params as netparams

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

nn=netparams.neural_network()

def plot_ablation_regions(ax, windows, color="lightcoral", alpha=0.25, label="ABLATION"):
    
    for start, end in windows:

        ax.axvspan(
            start,
            end,
            color=color,
            alpha=alpha
        )

        ax.text(
            (start + end) / 2,
            ax.get_ylim()[1] * 0.95,
            label,
            color="darkred",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )
        
def update_neuronal_characteristic(update_charac,neuron_population,leakage_value):
	neuron_charac = update_charac
	for neuron in neuron_population:
	    nest.SetStatus(neuron, {neuron_charac: leakage_value})
	new_val = nest.GetStatus(neuron_population, keys=neuron_charac)[0]
	return new_val

def read_recent_spike_data(spike_detector):
    total_spikes = 0
    spiketimes = [spike_detector.get('events', 'times')]
    spike_detection_window = 1 #ms
    current_time = nest.biological_time
    for i in range(len(spiketimes)):
        total_spikes += int(([np.sum((arr[i] >= current_time-spike_detection_window) & (arr[i] <= current_time)) for arr in spiketimes])[0])
    return total_spikes

def read_spike_data(spike_detector):
	senders = []
	spiketimes = []
	spike_detector = spike_detector
	senders += [spike_detector.get('events', 'senders')]
	spiketimes += [spike_detector.get('events', 'times')]
	return senders,spiketimes

def read_membrane_potential(multimeter,pop_size,neuron_num):
    mm = nest.GetStatus(multimeter,keys="events")[0]
    vm =  mm['V_m']
    t_vm = mm['times']
    vm = vm[neuron_num::pop_size]
    t_vm = t_vm[neuron_num::pop_size]
    
    # Ensure chronological order
    sorted_indices = np.argsort(t_vm)
    t_vm = t_vm[sorted_indices]
    vm = vm[sorted_indices]
    
    return vm,t_vm

def count_indiv_spikes(total_neurons,neuron_id_data,calc_freq):
        total_spikes_per_second = 6 if math.isnan(calc_freq) else int(calc_freq*2) #Spiking 2 times per period
        spike_count_array = [len(neuron_id_data[0][i]) for i in range(total_neurons)]
        sparse_count_max = total_spikes_per_second*(nn.sim_time/1000)	
        sparse_firing_count = [i for i, count in enumerate(spike_count_array) if count>=1 and count<=sparse_count_max]
        silent_neuron_count = [i for i, count in enumerate(spike_count_array) if count==0]
        neuron_to_sample = sparse_firing_count[1] if len(sparse_firing_count) > 1 else 0
        #print('Max for sparse firing for this trial: ',sparse_count_max)
        return spike_count_array,neuron_to_sample,len(sparse_firing_count),len(silent_neuron_count) 

def save_spike_data(num_neurons,population,neuron_num_offset):
	spike_time = []
	all_spikes = []
	for i in range(num_neurons):
	    spike_data = population[0][i]
	    neuron_num = [i+neuron_num_offset]*spike_data.shape[0]
	    for j in range(spike_data.shape[0]):
	        spike_time.append(spike_data[j])    
	    indiv_spikes = list(zip(neuron_num,spike_time))
	    all_spikes.extend(indiv_spikes)  
	    spike_time = []     
	return all_spikes

def single_neuron_spikes(neuron_number,population):
	spike_time = [0]*int(nn.sim_time/nn.time_resolution)
	spike_data = population[0][neuron_number]
	for j in range(spike_data.shape[0]):
	    spike_time_index = int(spike_data[j]*(1/nn.time_resolution))-1
	    spike_time[spike_time_index]=spike_data[j]        
	return spike_time

def single_neuron_spikes_binary(neuron_number, population):

    n_bins = int(nn.sim_time / nn.time_resolution)
    spike_time = np.zeros(n_bins, dtype=np.int8)

    # Neuron does not exist → silent
    if population is None or neuron_number >= len(population):
        return spike_time

    spike_data = population[neuron_number]

    if spike_data is None:
        return spike_time

    # ---- robust flattening ----
    if isinstance(spike_data, (list, tuple)):
        segments = []
        for seg in spike_data:
            if seg is None:
                continue
            arr = np.asarray(seg).ravel()
            if arr.size > 0:
                segments.append(arr)
        if segments:
            spike_data = np.concatenate(segments)
        else:
            spike_data = np.array([], dtype=float)

    elif isinstance(spike_data, np.ndarray):
        spike_data = spike_data.ravel()

    else:
        spike_data = np.array([], dtype=float)

    # ---- bin spikes ----
    for t in spike_data:
        if not np.isfinite(t):
            continue
        idx = int(t / nn.time_resolution)
        if 0 <= idx < n_bins:
            spike_time[idx] = 1

    return spike_time




def calculate_interspike_frequency(neuron_count, output_spiketimes):
    frequencies = []
    times = []

    # No spikes recorded at all
    if not output_spiketimes or len(output_spiketimes[0]) == 0:
        for _ in range(neuron_count):
            frequencies.append(np.array([np.nan]))
            times.append(np.array([np.nan]))
        return frequencies, times

    for i in range(neuron_count):

        # Neuron index does not exist → silent neuron
        if i >= len(output_spiketimes[0]):
            frequencies.append(np.array([np.nan]))
            times.append(np.array([np.nan]))
            continue

        t_spikes = np.asarray(output_spiketimes[0][i])

        # Need at least two spikes for ISI
        if t_spikes.size > 1:
            # Sort spikes by time
            spike_times = np.sort(t_spikes)

            isi = np.diff(spike_times)

            # Filter invalid ISIs
            valid_mask = np.isfinite(isi) & (isi > 0)
            valid_isi = isi[valid_mask]
            valid_times = spike_times[1:][valid_mask]

            if valid_isi.size > 0:
                frequencies.append(1000.0 / valid_isi)
                times.append(valid_times)
            else:
                frequencies.append(np.array([np.nan]))
                times.append(np.array([np.nan]))
        else:
            frequencies.append(np.array([np.nan]))
            times.append(np.array([np.nan]))

    return frequencies, times



def calculate_avg_interspike_frequencies(output_spiketimes):
    total_time = nn.sim_time
    bin_width = nn.time_window*nn.time_resolution
    bin_edges = np.arange(0, total_time + bin_width, bin_width)
    num_bins = len(bin_edges) - 1

    all_times = []
    all_freqs = []

    # Step 1–2: Collect all frequencies and their associated times
    for neuron_spikes in output_spiketimes[0]:
        if len(neuron_spikes) > 1:
            sorted_spikes = np.sort(neuron_spikes)
            isi = np.diff(sorted_spikes)
            freqs = 1000.0 / isi  # Hz
            times = sorted_spikes[1:]  # Time of the second spike in the ISI
            all_freqs.extend(freqs)
            all_times.extend(times)

    all_freqs = np.array(all_freqs)
    all_times = np.array(all_times)

    # Step 3: Bin the data
    bin_sums = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    bin_indices = np.digitize(all_times, bin_edges) - 1
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < num_bins:
            bin_sums[bin_idx] += all_freqs[i]
            bin_counts[bin_idx] += 1

    # Step 4: Compute averages
    with np.errstate(invalid='ignore'):
        avg_freqs = np.divide(bin_sums, bin_counts, where=bin_counts != 0)

    # Step 5: Fill empty bins by averaging neighbors
    for i in range(num_bins):
        if bin_counts[i] == 0:
            prev_val = next_val = None
            # Search left
            for j in range(i - 1, -1, -1):
                if bin_counts[j] != 0:
                    prev_val = avg_freqs[j]
                    break
            # Search right
            for j in range(i + 1, num_bins):
                if bin_counts[j] != 0:
                    next_val = avg_freqs[j]
                    break
            if prev_val is not None and next_val is not None:
                avg_freqs[i] = (prev_val + next_val) / 2
            elif prev_val is not None:
                avg_freqs[i] = prev_val
            elif next_val is not None:
                avg_freqs[i] = next_val
            else:
                avg_freqs[i] = 0  # No data at all

    smoothed_freqs = gaussian_filter(avg_freqs, 2)

    return smoothed_freqs, bin_edges[:-1]


def padded_sliding_time_window(signal, window_size):
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded_signal, window_size)
    return np.mean(windows, axis=1)[:len(signal)]  # Trim to original size


def rate_code_spikes(neuron_count, output_spiketimes):
    # Time bins
    bins = np.arange(
        0,
        nn.sim_time + nn.time_resolution,
        nn.time_resolution
    )

    # Initialize spike bins to zero
    spike_bins_current = np.zeros(len(bins) - 1)

    # No spikes recorded at all
    if not output_spiketimes or len(output_spiketimes[0]) == 0:
        # Still apply smoothing pipeline for consistency
        spike_bins_current = sliding_time_window(
            spike_bins_current, nn.time_window
        )
        smoothed_spike_bins = gaussian_filter(
            spike_bins_current, nn.convstd_rate
        )
        if nn.chop_edges_amount > 0.0:
            smoothed_spike_bins = smoothed_spike_bins[
                int(nn.chop_edges_amount) : int(-nn.chop_edges_amount)
            ]
        return smoothed_spike_bins

    # Loop over neurons safely
    for i in range(neuron_count):

        # If this neuron index does not exist → silent neuron
        if i >= len(output_spiketimes[0]):
            continue

        t_spikes = output_spiketimes[0][i]

        # Convert scalar to array if necessary
        if np.isscalar(t_spikes):
            t_spikes = np.array([t_spikes])
        else:
            t_spikes = np.array(t_spikes)

        # Skip if empty
        if t_spikes.size == 0:
            continue

    
        # Histogram spike times
        spikes_per_bin, _ = np.histogram(t_spikes, bins)

        spike_bins_current += spikes_per_bin

    # Smooth in time
    spike_bins_current = sliding_time_window(
        spike_bins_current, nn.time_window
    )
    smoothed_spike_bins = gaussian_filter(
        spike_bins_current, nn.convstd_rate
    )

    if nn.chop_edges_amount > 0.0:
        smoothed_spike_bins = smoothed_spike_bins[
            int(nn.chop_edges_amount) : int(-nn.chop_edges_amount)
        ]

    return smoothed_spike_bins


def sliding_time_window(signal, window_size):
	windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
	return np.sum(windows, axis=1)

def sliding_time_window_matrix(signal, window_size):
	result = []
	for row in signal:
	    windows = np.lib.stride_tricks.sliding_window_view(row, window_size)
	    row_sum = np.sum(windows, axis=1)
	    result.append(row_sum)
	return np.array(result)

def smooth(data, sd):
	data = copy.copy(data)
	from scipy.signal import convolve, windows
	n_bins = data.shape[1]
	w = n_bins - 1 if n_bins % 2 == 0 else n_bins
	window = windows.gaussian(w, std=sd)
	for j in range(data.shape[0]):
	    data[j,:] = convolve(data[j,:], window, mode='same', method='auto') 
	return data

def convolve_spiking_activity(population_size,population):
    
    spike_times = population 

    # unwrapping block 
    if (
        isinstance(spike_times, list)
        and len(spike_times) == 1
        and isinstance(spike_times[0], (list, np.ndarray))
        and len(spike_times[0]) == population_size
    ):
        spike_times = spike_times[0]

    #print("[DEBUG] convolve_spiking_activity:", len(spike_times))

    time_steps = int(nn.sim_time/nn.time_resolution) 
    
    # calling the single_neuron_spikes_binary - error taking place here, calling the binary times here. 
    binary_spikes = np.vstack([
    single_neuron_spikes_binary(i, spike_times)
    for i in range(population_size)
    ])
    
    # binning spikes for convolving 
    #binned_spikes = sliding_time_window_matrix(binary_spikes,nn.time_window)
    binned_spikes = binary_spikes # per becks commit. 

    smoothed_spikes = smooth(binned_spikes, nn.convstd_rate)

    time_vector = np.arange(binned_spikes.shape[1]) * nn.time_resolution

    # chop edges amount paramter 
    if nn.chop_edges_amount > 0.0:
        chop = int(nn.chop_edges_amount)
        smoothed_spikes = smoothed_spikes[:, chop:-chop]
        time_vector = time_vector[chop:-chop]

    # remove mean parameter 
    if nn.remove_mean:
        smoothed_spikes = (smoothed_spikes.T - np.mean(smoothed_spikes, axis=1)).T

    # high pass filtered parameter 
    if nn.high_pass_filtered:
        # Same used as in Linden et al, 2022 paper
        b, a = butter(3, .1, 'highpass', fs=1000)		#high pass freq was previously 0.3Hz
        smoothed_spikes = filtfilt(b, a, smoothed_spikes)
    
    # downsampling convolved parameter 
    if nn.downsampling_convolved:
        decimation_factor = int(1 / nn.time_resolution)  
        smoothed_spikes = decimate(
            smoothed_spikes,
            decimation_factor,
            n=2,
            ftype='iir',
            zero_phase=True
        )
        time_vector = time_vector[::decimation_factor]
    
    smoothed_spikes = smoothed_spikes[:, :-nn.time_window + 1] #truncate array by the width of the time window 
    
    time_vector = time_vector[:smoothed_spikes.shape[1]]
    pop_mean = smoothed_spikes.mean(axis=0)
    
    return pop_mean, time_vector, smoothed_spikes


def inject_current(neuron_population,current):
	for neuron in neuron_population:
	    nest.SetStatus([neuron],{"I_e": current})
	updated_current = nest.GetStatus(neuron_population, keys="I_e")[0]
	return updated_current
	
def normalize_rows(matrix):
    max_values = np.max(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / max_values
    return normalized_matrix	

def spike_report(name, senders, spiketimes):
    """
    Fast debug report: confirms spiking is happening and gives quick counts.
    Accepts common formats:
      - spiketimes[0] = list of arrays/lists (one per neuron)  [your current assumption]
      - spiketimes = list of arrays/lists (one per neuron)
      - spiketimes = flat array of spike times (single list)
    """
    print(f"\n===== SPIKE SANITY CHECK : {name} =====")

    if spiketimes is None:
        print("No spiketimes provided (None).")
        return

    # --- normalize trains into a list of per-neuron spike arrays ---
    trains = None

    try:
        # Case 1: your expected structure: spiketimes[0] is list-of-trains
        if isinstance(spiketimes, (list, tuple)) and len(spiketimes) > 0:
            if isinstance(spiketimes[0], (list, tuple, np.ndarray)):
                # If spiketimes[0] itself looks like a list-of-trains (list of lists/arrays)
                # We detect by checking whether it’s a list where elements are themselves list/array of numbers
                if (len(spiketimes) == 1) and isinstance(spiketimes[0], (list, tuple)) and \
                   (len(spiketimes[0]) > 0) and isinstance(spiketimes[0][0], (list, tuple, np.ndarray)):
                    trains = spiketimes[0]
                # If spiketimes looks like list-of-trains already
                elif len(spiketimes) > 0 and isinstance(spiketimes[0], (list, tuple, np.ndarray)) and \
                     (len(spiketimes[0]) == 0 or np.isscalar(np.array(spiketimes[0]).ravel()[0])):
                    # Could be list-of-trains OR a single flat list.
                    # Decide: if elements are arrays/lists -> list-of-trains, else flat list
                    if len(spiketimes) > 0 and isinstance(spiketimes[0], (list, tuple, np.ndarray)) and \
                       (len(spiketimes[0]) > 0) and not isinstance(spiketimes[0][0], (list, tuple, np.ndarray)):
                        # This is ambiguous: it might be a single train represented as list of times
                        # Treat it as ONE neuron train.
                        trains = [np.asarray(spiketimes, dtype=float)]
                    else:
                        trains = spiketimes

        # Case 2: spiketimes is a numpy array of times => treat as one train
        if trains is None and isinstance(spiketimes, np.ndarray):
            trains = [spiketimes.astype(float)]

    except Exception as e:
        print("Error while normalizing spiketimes:", e)
        print("Raw spiketimes:", spiketimes)
        return

    if trains is None or len(trains) == 0:
        print("No spike trains found / empty spiketimes.")
        return

    # --- compute counts safely ---
    try:
        counts = []
        all_spikes = []

        for st in trains:
            arr = np.asarray(st, dtype=float).ravel()
            counts.append(arr.size)
            if arr.size:
                all_spikes.append(arr)

        total_neurons = len(trains)
        total_spikes = int(np.sum(counts))
        active_neurons = int(np.sum(np.array(counts) > 0))
        silent_neurons = total_neurons - active_neurons

    except Exception as e:
        print("Error while parsing spike trains:", e)
        print("Raw trains:", trains)
        return

    print(f"Total neurons: {total_neurons}")
    print(f"Active neurons: {active_neurons}")
    print(f"Silent neurons: {silent_neurons}")
    print(f"Total spikes: {total_spikes}")

    if total_spikes == 0:
        print("No spikes detected — population is SILENT.")
        return

    # --- tiny extra debug info (still fast) ---
    try:
        flat = np.concatenate(all_spikes) if all_spikes else np.array([])
        if flat.size:
            t0 = float(np.min(flat))
            t1 = float(np.max(flat))
            dur_ms = max(1.0, (t1 - t0))
            mean_rate_hz = total_spikes / (total_neurons * (dur_ms / 1000.0))
            print(f"Spike window: {t0:.1f}–{t1:.1f} ms (Δ={dur_ms:.1f} ms)")
            print(f"Mean rate (rough): {mean_rate_hz:.2f} Hz/neuron")
    except Exception:
        # If anything goes wrong here, don’t break debugging output
        pass

    print("Spiking OK. Moving onto in-depth analysis... ")
    time.sleep(1)


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


def _dt_ms(t_ms: np.ndarray) -> float:
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size < 2:
        return np.nan
    d = np.diff(t_ms)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    return float(np.median(d))


def compute_signal_metrics(
    t_ms: np.ndarray,
    y: np.ndarray,
    *,
    thresh_frac: float = 0.2,
    min_peak_distance_ms: float = 200.0,
):
    """
    Compute burst/cycle metrics from a (scaled) convolved firing-rate signal.

    Returns a dict with:
      mean_hz, max_hz, amp_hz,
      n_peaks,
      cycle_period_ms, cycle_freq_hz,
      burst_duration_ms, duty_cycle,
      active_ms
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    out = {
        "mean_hz": np.nan,
        "max_hz": np.nan,
        "amp_hz": np.nan,
        "n_peaks": 0,
        "cycle_period_ms": np.nan,
        "cycle_freq_hz": np.nan,
        "burst_duration_ms": np.nan,
        "duty_cycle": np.nan,
        "active_ms": np.nan,
    }

    if t_ms.size == 0 or y.size == 0:
        return out

    # Basic stats
    y_f = y[np.isfinite(y)]
    if y_f.size == 0:
        return out

    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    out["mean_hz"] = float(np.nanmean(y))
    out["max_hz"] = y_max
    out["amp_hz"] = y_max - y_min

    dt = _dt_ms(t_ms)
    if not np.isfinite(dt) or dt <= 0:
        return out

    # Peak detection
    dist_samp = max(1, int(round(min_peak_distance_ms / dt)))
    peaks, props = find_peaks(y, distance=dist_samp)

    out["n_peaks"] = int(peaks.size)
    if peaks.size >= 2:
        peak_times = t_ms[peaks]
        periods = np.diff(peak_times)
        periods = periods[np.isfinite(periods) & (periods > 0)]
        if periods.size:
            cycle_period_ms = float(np.mean(periods))
            out["cycle_period_ms"] = cycle_period_ms
            out["cycle_freq_hz"] = 1000.0 / cycle_period_ms if cycle_period_ms > 0 else np.nan

    # Burst threshold mask (relative to max)
    thr = thresh_frac * y_max
    active = np.asarray(y >= thr, dtype=bool)

    out["active_ms"] = float(np.sum(active) * dt)

    # Burst durations (contiguous active segments)
    # Find rising/falling edges
    if active.size >= 2:
        edges = np.diff(active.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1

        # If starts after an active region began
        if active[0]:
            starts = np.r_[0, starts]
        # If ends missing because active at end
        if active[-1]:
            ends = np.r_[ends, active.size]

        if starts.size and ends.size and starts.size == ends.size:
            durs = (ends - starts) * dt
            durs = durs[np.isfinite(durs) & (durs > 0)]
            if durs.size:
                out["burst_duration_ms"] = float(np.mean(durs))

    # Duty cycle: burst_duration / cycle_period
    if np.isfinite(out["burst_duration_ms"]) and np.isfinite(out["cycle_period_ms"]) and out["cycle_period_ms"] > 0:
        out["duty_cycle"] = out["burst_duration_ms"] / out["cycle_period_ms"]

    return out


def compute_pair_metrics(
    t_ms: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    *,
    thresh_frac: float = 0.2,
):
    """
    Pairwise alternation/overlap metrics computed from thresholded activity.
    Returns:
      overlap_ms, overlap_frac_total, alt_index
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)

    out = {
        "overlap_ms": np.nan,
        "overlap_frac_total": np.nan,
        "alt_index": np.nan,
    }

    if t_ms.size == 0 or y_a.size == 0 or y_b.size == 0:
        return out

    dt = _dt_ms(t_ms)
    if not np.isfinite(dt) or dt <= 0:
        return out

    thr_a = thresh_frac * float(np.nanmax(y_a)) if np.isfinite(np.nanmax(y_a)) else np.nan
    thr_b = thresh_frac * float(np.nanmax(y_b)) if np.isfinite(np.nanmax(y_b)) else np.nan

    a = (y_a >= thr_a) if np.isfinite(thr_a) else np.zeros_like(y_a, dtype=bool)
    b = (y_b >= thr_b) if np.isfinite(thr_b) else np.zeros_like(y_b, dtype=bool)

    overlap = a & b
    out["overlap_ms"] = float(np.sum(overlap) * dt)

    total_ms = float((t_ms[-1] - t_ms[0]) if t_ms.size >= 2 else (t_ms.size * dt))
    if total_ms > 0:
        out["overlap_frac_total"] = out["overlap_ms"] / total_ms

    a_ms = float(np.sum(a) * dt)
    b_ms = float(np.sum(b) * dt)

    denom = (a_ms + b_ms)
    if denom > 0:
        # 1 = perfect alternation (no overlap), 0 = complete overlap
        out["alt_index"] = 1.0 - (2.0 * out["overlap_ms"] / denom)

    return out


def export_metrics_table_clean(
    out_csv_path: str,
    *,
    label: str,
    t_ms: np.ndarray,
    populations: dict,
    existing: dict | None = None,
    meta: dict | None = None,
    thresh_frac: float = 0.2,
    min_peak_distance_ms: float = 200.0,
    round_ndp: int = 4,
):
    """
    populations format:
      {
        "MNP_ipsilateral": {"signals": {"FLX": y1, "EXT": y2}, "pairs":[("FLX","EXT")]},
        ...
      }

    Writes a tidy CSV (and returns DataFrame).
    """
    existing = existing or {}
    meta = meta or {}

    rows = []
    t_ms = np.asarray(t_ms, dtype=float)

    for pop_name, pop_spec in populations.items():
        signals = pop_spec.get("signals", {})
        pairs = pop_spec.get("pairs", [])

        # --- per-signal metrics ---
        sig_metrics = {}
        for sig_name, y in signals.items():
            m = compute_signal_metrics(
                t_ms, y,
                thresh_frac=thresh_frac,
                min_peak_distance_ms=min_peak_distance_ms,
            )
            for k, v in m.items():
                sig_metrics[f"{sig_name}_{k}"] = _safe_float(v)

        # --- per-pair metrics ---
        pair_metrics = {}
        for (a, b) in pairs:
            if a in signals and b in signals:
                pm = compute_pair_metrics(t_ms, signals[a], signals[b], thresh_frac=thresh_frac)
                pair_metrics[f"{a}_vs_{b}_overlap_ms"] = _safe_float(pm["overlap_ms"])
                pair_metrics[f"{a}_vs_{b}_overlap_frac_total"] = _safe_float(pm["overlap_frac_total"])
                pair_metrics[f"{a}_vs_{b}_alt_index"] = _safe_float(pm["alt_index"])

        row = {
            "label": label,
            "population": pop_name,
            **meta,
            **existing,
            **sig_metrics,
            **pair_metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Clean formatting
    for c in df.columns:
        if df[c].dtype.kind in ("f", "i"):
            df[c] = pd.to_numeric(df[c], errors="coerce").round(round_ndp)

    # Prefer meta columns first (if present)
    front = ["label", "population"]
    for k in (meta.keys() if isinstance(meta, dict) else []):
        if k in df.columns and k not in front:
            front.append(k)

   #  Then existing metrics
    for k in (existing.keys() if isinstance(existing, dict) else []):
        if k in df.columns and k not in front:
            front.append(k)

    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    df.to_csv(out_csv_path, index=False)
    return df


def window_rates_from_spike_detector(
    spike_detector,
    n_neurons: int,
    t0_ms: float,
    t1_ms: float,
    return_spike_times=True,
):
    """
    Returns:
      - n_spikes
      - pop_hz
      - mean_hz
      - spike_times_ms (exact spike times in window, concatenated)
    
    Works with:
      - a single NEST spike_recorder
      - a NodeCollection of spike_recorders
    """
    events_list = nest.GetStatus(spike_detector, "events")

    spike_times_all = []
    n_spikes = 0

    for ev in events_list:
        times = np.asarray(ev.get("times", []), dtype=float)
        if times.size == 0:
            continue

        # restrict to window
        m = (times >= t0_ms) & (times < t1_ms)
        tw = times[m]

        if tw.size > 0:
            spike_times_all.append(tw)
            n_spikes += int(tw.size)

    if spike_times_all:
        spike_times_ms = np.concatenate(spike_times_all)
        spike_times_ms.sort()
    else:
        spike_times_ms = np.array([], dtype=float)

    dur_s = max(1e-12, (float(t1_ms) - float(t0_ms)) / 1000.0)
    n_neurons = max(1, int(n_neurons))

    pop_hz = n_spikes / dur_s
    mean_hz = pop_hz / n_neurons

    out = {
        "n_spikes": int(n_spikes),
        "dur_ms": float(t1_ms - t0_ms),
        "pop_hz": float(pop_hz),
        "mean_hz": float(mean_hz),
    }

    if return_spike_times:
        out["spike_times_ms"] = spike_times_ms

    return out


def window_peak_rate_metrics(
    t_ms: np.ndarray,
    y: np.ndarray,
    t0_ms: float,
    t1_ms: float,
    min_peak_height: float = 100.0,
    min_peak_distance_ms: float = 200.0,
):
    """
    Compute simple burst metrics from a rate-coded (convolved) signal y(t):
      - n_peaks
      - peak_rate_hz  (peaks per second)

    Notes:
    - Uses a very lightweight local-maximum detector (no scipy dependency).
    - min_peak_distance_ms enforces a refractory between detected peaks.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    if t_ms.size == 0 or y.size == 0 or t_ms.size != y.size:
        return {"n_peaks": 0, "peak_rate_hz": 0.0, "peak_times": []}

    m = (t_ms >= float(t0_ms)) & (t_ms <= float(t1_ms))
    tt = t_ms[m]
    yy = y[m]

    if tt.size < 3:
        return {"n_peaks": 0, "peak_rate_hz": 0.0, "peak_times": []}

    # local maxima: yy[i-1] < yy[i] >= yy[i+1] and above threshold
    cand = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] >= yy[2:]) & (yy[1:-1] >= float(min_peak_height)))[0] + 1
    if cand.size == 0:
        return {"n_peaks": 0, "peak_rate_hz": 0.0, "peak_times": []}

    # enforce minimum time distance between peaks (greedy)
    peak_times = []
    last_t = -np.inf
    for idx in cand:
        ti = float(tt[idx])
        if (ti - last_t) >= float(min_peak_distance_ms):
            peak_times.append(ti)
            last_t = ti

    dur_s = max(1e-12, (float(t1_ms) - float(t0_ms)) / 1000.0)
    n_peaks = int(len(peak_times))
    peak_rate_hz = float(n_peaks / dur_s)

    return {
        "n_peaks": n_peaks,
        "peak_rate_hz": peak_rate_hz,
        "peak_times": peak_times,
    }

def collapsed_by_peaks(t_ms, y, t0_ms, t1_ms, min_peak_height=100.0, min_peaks=1, min_peak_distance_ms=200.0):
    """
    Collapse if fewer than min_peaks peaks above threshold in the plateau window.
    """
    pm = window_peak_rate_metrics(
        t_ms, y, t0_ms, t1_ms,
        min_peak_height=min_peak_height,
        min_peak_distance_ms=min_peak_distance_ms
    )
    collapsed = (pm["n_peaks"] < int(min_peaks))
    pm["collapsed"] = bool(collapsed)
    return collapsed, pm


# ======================================
# BIFURICATION ANALYSIS HELPER FUNCTIONS

def get_baseline_weights(network, weight_key):
    """
    Extract baseline weight for a synapse type.
    
    Parameters
    ----------
    network : dict
        Network object with 'connections' and parameters
    weight_key : str
        Key for synapse type (e.g., 'v0d_to_rg_inh')
    
    Returns
    -------
    w0 : float
        Baseline weight in nanosiemens or pA
    """
    # Get from network config - adjust based on your parameter names
    # Example: w0 = network.get('wcustomv0drginhL', -0.5)
    # For now, placeholder - you'll fill in with actual baseline
    w0 = -0.5  # V0D -> RG_L inhibition baseline (pA or nS)
    return w0


def compute_pop_rate(spike_detector, n_neurons, bin_width_ms=50):
    """
    Compute population firing rate from spike detector.
    
    Parameters
    ----------
    spike_detector : NEST gid
        Spike detector node
    n_neurons : int
        Number of neurons in population
    bin_width_ms : float
        Binning window (ms)
    
    Returns
    -------
    rate : float
        Firing rate (Hz) in last bin
    """
    events = nest.GetStatus(spike_detector, 'events')[0]
    spike_times = events['times']
    
    if len(spike_times) == 0:
        return 0.0
    
    # Get rate in most recent bin
    t_end = nest.biological_time
    t_start = t_end - bin_width_ms
    spikes_in_bin = np.sum((spike_times >= t_start) & (spike_times <= t_end))
    
    # Normalize: spikes per bin / (n_neurons Ã— bin_width_seconds)
    rate_hz = (spikes_in_bin / n_neurons) / (bin_width_ms / 1000.0)
    
    return rate_hz

def record_slow_variables(network, current_time_ms, n_rg_l, n_rg_r, n_v0d, n_v1a):
    """
    Extract all slow variables from spike detectors.
    
    Returns
    -------
    record : dict
        {time_ms, r_RG_L, r_RG_R, v_V0D_L, v_V0D_R, v_V1a, is_active}
    """
    record = {
        'time_ms': current_time_ms,
        'r_RG_L': compute_pop_rate(network['detectors']['RG_L'], n_rg_l, bin_width_ms=50),
        'r_RG_R': compute_pop_rate(network['detectors']['RG_R'], n_rg_r, bin_width_ms=50),
        'v_V0D_L': compute_pop_rate(network['detectors']['V0D_L'], n_v0d, bin_width_ms=50),
        'v_V0D_R': compute_pop_rate(network['detectors']['V0D_R'], n_v0d, bin_width_ms=50),
        'v_V1a': compute_pop_rate(network['detectors']['V1a'], n_v1a, bin_width_ms=50),
    }
    
    # Simple activity criterion: RG firing > 1 Hz
    is_active = (record['r_RG_L'] > 1.0) or (record['r_RG_R'] > 1.0)
    record['is_active'] = is_active
    
    return record

def apply_y_ramp(network, connections_key, current_progress, config):
    """
    Update weights to implement Y ramp.
    
    Parameters
    ----------
    network : dict
        Network connections
    connections_key : str
        Which synapses to ramp (e.g., 'v0d_to_rg_inh')
    current_progress : float
        Normalized progress [0, 1]
    config : RampConfig
    """
    # Get baseline weight
    w0 = get_baseline_weights(network, connections_key)
    
    # Linear ramp: multiply from start_mult to end_mult
    mult = config.y_start_mult + (config.y_end_mult - config.y_start_mult) * current_progress
    w_new = w0 * mult
    
    # Apply to connections
    connections = network['connections'].get(connections_key, [])
    if connections:
        nest.SetStatus(connections, [{'weight': float(w_new)} for _ in connections])
    
    return w_new 

def mult_from_config(val):
    # val is expected like 0, -1, -2, -3 ...
    # 0 -> 1.0, -3 -> 3.0
    return 1.0 + abs(float(val))


def robust_z(x, eps=1e-12):
    """
    Robust z-score using median/MAD.
    Returns z units that are stable across amplitude scaling.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad + eps
    return (x - med) / scale

def window_peak_metrics(
    t_ms: np.ndarray,
    y: np.ndarray,
    t0_ms: float,
    t1_ms: float,
    *,
    min_peak_distance_ms: float = 200.0,
    # adaptive mode (recommended)
    adaptive: bool = True,
    z_min_height: float = 1.0,
    z_min_prom: float = 0.5,
    # absolute fallback mode
    min_peak_height: float | None = None,
    min_prominence: float | None = None,
):
    """
    Peak metrics in a time window. Returns:
      n_peaks, peak_times, ipi_ms, median_ipi_ms, cv_ipi,
      peak_heights, prominence, prominence_mean
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    out = {
        "n_peaks": 0,
        "peak_times": np.array([], dtype=float),
        "ipi_ms": np.array([], dtype=float),
        "median_ipi_ms": np.nan,
        "cv_ipi": np.nan,
        "peak_heights": np.array([], dtype=float),
        "prominence": np.array([], dtype=float),
        "prominence_mean": np.nan,
    }

    if t_ms.size == 0 or y.size == 0 or t_ms.size != y.size:
        return out

    m = (t_ms >= float(t0_ms)) & (t_ms <= float(t1_ms))
    tt = t_ms[m]
    yy = y[m]
    if tt.size < 3:
        return out

    dt = _dt_ms(tt)
    if not np.isfinite(dt) or dt <= 0:
        return out
    dist_samp = max(1, int(round(min_peak_distance_ms / dt)))

    if adaptive:
        zz = robust_z(yy)
        peaks, props = find_peaks(zz, distance=dist_samp, height=z_min_height, prominence=z_min_prom)
        peak_times = tt[peaks]
        peak_heights = yy[peaks]  # report in original units
        prom = props.get("prominences", np.array([], dtype=float))
    else:
        # absolute thresholding
        h = float(min_peak_height) if min_peak_height is not None else float(np.nanpercentile(yy, 75))
        p = float(min_prominence) if min_prominence is not None else 0.2 * float(np.nanmax(yy) - np.nanmin(yy))
        peaks, props = find_peaks(yy, distance=dist_samp, height=h, prominence=p)
        peak_times = tt[peaks]
        peak_heights = yy[peaks]
        prom = props.get("prominences", np.array([], dtype=float))

    out["n_peaks"] = int(peaks.size)
    out["peak_times"] = peak_times
    out["peak_heights"] = peak_heights
    out["prominence"] = prom
    out["prominence_mean"] = float(np.nanmean(prom)) if prom.size else np.nan

    if peak_times.size >= 2:
        ipi = np.diff(peak_times)
        ipi = ipi[np.isfinite(ipi) & (ipi > 0)]
        out["ipi_ms"] = ipi
        if ipi.size:
            out["median_ipi_ms"] = float(np.nanmedian(ipi))
            mu = float(np.nanmean(ipi))
            sd = float(np.nanstd(ipi, ddof=1)) if ipi.size > 1 else 0.0
            out["cv_ipi"] = float(sd / mu) if mu > 0 else np.nan

    return out

def rg_regime_metrics_over_windows(
    t_ms: np.ndarray,
    rgF: np.ndarray,
    rgE: np.ndarray,
    *,
    check_window_ms: float = 5000.0,
    step_ms: float | None = None,   # None => non-overlapping
    warmup_ms: float = 50.0,
    thresh_frac: float = 0.15,
    min_peak_distance_ms: float = 500.0,
    adaptive_peaks: bool = True,
    z_min_height: float = 0.5,
    z_min_prom: float = 0.25,
):
    t_ms = np.asarray(t_ms, dtype=float)
    if step_ms is None:
        step_ms = check_window_ms

    t0 = float(np.nanmin(t_ms) + warmup_ms)
    t1 = float(np.nanmax(t_ms))

    windows = []
    cur = t0
    while cur + check_window_ms <= t1:
        windows.append((cur, cur + check_window_ms))
        cur += step_ms

    out = []
    for (w0, w1) in windows:
        # Restrict to window once
        m = (t_ms >= w0) & (t_ms <= w1)
        tt = t_ms[m]
        if tt.size < 3:
            continue

        yF = np.asarray(rgF, dtype=float)[m]
        yE = np.asarray(rgE, dtype=float)[m]

        # reuse your existing signal metrics
        sigF = compute_signal_metrics(tt, yF, thresh_frac=thresh_frac, min_peak_distance_ms=min_peak_distance_ms)
        sigE = compute_signal_metrics(tt, yE, thresh_frac=thresh_frac, min_peak_distance_ms=min_peak_distance_ms)

        # add IPI/prominence metrics (missing piece)
        pkF = window_peak_metrics(
            tt, yF, float(tt[0]), float(tt[-1]),
            min_peak_distance_ms=min_peak_distance_ms,
            adaptive=adaptive_peaks,
            z_min_height=z_min_height,
            z_min_prom=z_min_prom,
        )
        pkE = window_peak_metrics(
            tt, yE, float(tt[0]), float(tt[-1]),
            min_peak_distance_ms=min_peak_distance_ms,
            adaptive=adaptive_peaks,
            z_min_height=z_min_height,
            z_min_prom=z_min_prom,
        )

        out.append({
            "window": (w0, w1),
            "RG_F": {**sigF,
                     "median_ipi_ms": pkF["median_ipi_ms"],
                     "cv_ipi": pkF["cv_ipi"],
                     "prominence_mean": pkF["prominence_mean"],
                     "n_peaks_peaks": pkF["n_peaks"]},
            "RG_E": {**sigE,
                     "median_ipi_ms": pkE["median_ipi_ms"],
                     "cv_ipi": pkE["cv_ipi"],
                     "prominence_mean": pkE["prominence_mean"],
                     "n_peaks_peaks": pkE["n_peaks"]},
        })

    return out

def calibrate_rg_regime_from_healthy(
    healthy_runs: list,
    *,
        # how strict is "healthy"? (99th percentile) we were finding too many classifications with 90th percentile? 
        upper_q: float = 0.995,
        lower_q: float = 0.005,
        check_window_ms: float = 2000.0,
        bin_ms: float = 200.0,
    ):
    """
    healthy_runs: list of dicts, each dict contains:
      {
        "t_ms": t,
        "rgF_ipsi": y, "rgE_ipsi": y,
        "rgF_contra": y, "rgE_contra": y
      }
    Returns a recommended config dict.
    """
    all_npeaks = []
    all_med_ipi = []
    all_cv = []
    all_prom = []

    for run in healthy_runs:
        t = run["t_ms"]

        for side in ("ipsi", "contra"):
            metrics = rg_regime_metrics_over_windows(
                t, run[f"rgF_{side}"], run[f"rgE_{side}"],
                check_window_ms=check_window_ms,
                min_peak_distance_ms=200.0,  # initial; you can refine after first pass
                adaptive_peaks=True,
            )
            for w in metrics:
                for k in ("RG_F", "RG_E"):
                    m = w[k]
                    all_npeaks.append(m.get("n_peaks_peaks", np.nan))
                    all_med_ipi.append(m.get("median_ipi_ms", np.nan))
                    all_cv.append(m.get("cv_ipi", np.nan))
                    all_prom.append(m.get("prominence_mean", np.nan))

    def finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    npeaks = finite(all_npeaks)
    med_ipi = finite(all_med_ipi)
    cv = finite(all_cv)
    prom = finite(all_prom)

    if npeaks.size == 0 or med_ipi.size == 0:
        raise ValueError("Healthy calibration failed: no peaks/IPIs found. Check scaling or adaptive z params.")

    # MIN_PEAKS: conservative minimum number of significant peaks per window
    MIN_PEAKS = int(max(1, np.floor(np.quantile(npeaks, lower_q))))

    # MAX_MEDIAN_IPI_MS: upper bound on typical period
    MAX_MEDIAN_IPI_MS = float(np.quantile(med_ipi, upper_q))

    # MAX_CV: upper bound on irregularity
    MAX_CV = float(np.quantile(cv, upper_q)) if cv.size else np.nan

    # MIN_PEAK_DIST_MS: set as a fraction of lower-tail healthy IPI (avoid double counting)
    min_healthy_ipi = float(np.quantile(med_ipi, lower_q))
    MIN_PEAK_DIST_MS = float(max(1.0, 0.5 * min_healthy_ipi))

    # PROMINENCE: only meaningful if you later choose absolute prominence.
    # If you stick with adaptive z-prominence, you can store the recommended *absolute* as info.
    PROMINENCE = float(np.quantile(prom, lower_q)) if prom.size else np.nan

    return {
        "BIN_MS": float(bin_ms),
        "CHECK_WINDOW_MS": float(check_window_ms),
        "MIN_PEAKS": int(MIN_PEAKS),
        "MIN_PEAK_DIST_MS": float(MIN_PEAK_DIST_MS),
        "MAX_MEDIAN_IPI_MS": float(MAX_MEDIAN_IPI_MS),
        "MAX_CV": float(MAX_CV),
        "PROMINENCE": float(PROMINENCE),
        # plus: recommended to store the adaptive params you used
        "ADAPTIVE_PEAKS": True,
        "Z_MIN_HEIGHT": 1.0,
        "Z_MIN_PROM": 0.5,
    }


def resolve_targets_to_nc(target_strs, *, roots: dict):
    """
    Resolve config strings like "L_V0D.v0d_tonic" into a single NodeCollection.
    Safe: only allows traversal from explicitly provided `roots`.
    """
    if not target_strs:
        raise ValueError("[ERROR] target_strs is empty.")

    out = None

    for s in target_strs:
        if not isinstance(s, str) or "." not in s:
            raise TypeError(f"[ERROR] Bad target '{s}'. Expected string like 'L_V0D.v0d_tonic'.")

        head, *attrs = s.split(".")
        if head not in roots:
            raise KeyError(f"[ERROR] Unknown root '{head}' in '{s}'. Allowed roots: {list(roots.keys())}")

        obj = roots[head]
        for a in attrs:
            if not hasattr(obj, a):
                raise AttributeError(f"[ERROR] '{s}' failed: '{type(obj).__name__}' has no attr '{a}'")
            obj = getattr(obj, a)

        if not isinstance(obj, nest.NodeCollection):
            raise TypeError(f"[ERROR] '{s}' did not resolve to NodeCollection. Got: {type(obj)}")

        out = obj if out is None else (out + obj)

    if out is None or len(out) == 0:
        raise ValueError("[ERROR] Resolved drive_targets is empty.")

    return out

def plateau_peak_metrics(t_ms, y, t0, t1,
                         min_peak_distance_ms,
                         prominence):
    """Compute peaks + IPI stats inside [t0,t1]."""
    # window mask
    m = (t_ms >= t0) & (t_ms <= t1)
    tw = t_ms[m]
    yw = y[m]

    if tw.size < 3:
        return {
            "n_peaks": 0,
            "peak_times": np.array([]),
            "median_ipi_ms": np.nan,
            "cv_ipi": np.nan,
        }

    # estimate sampling dt to convert distance(ms)->samples
    dt = float(np.median(np.diff(tw)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    min_dist_samp = int(max(1, round(min_peak_distance_ms / dt)))

    # peaks
    peaks, props = find_peaks(yw, distance=min_dist_samp, prominence=float(prominence))
    peak_times = tw[peaks] if peaks.size else np.array([])

    # IPI stats
    if peak_times.size >= 2:
        ipis = np.diff(peak_times)  # ms
        med_ipi = float(np.median(ipis))
        mu = float(np.mean(ipis))
        sd = float(np.std(ipis))
        cv = float(sd / mu) if mu > 0 else np.nan
    else:
        med_ipi = np.nan
        cv = np.nan

    return {
        "n_peaks": int(peak_times.size),
        "peak_times": peak_times,
        "median_ipi_ms": med_ipi,
        "cv_ipi": cv,
    }


def detect_peaks_simple(t_ms, y, t0_ms, t1_ms, min_peak_prom=None, min_peak_dist_ms=None):
    """
    Simple peak detector without scipy.
    Uses local maxima + optional prominence-ish threshold + optional min distance.

    Args:
      t_ms: 1D array times in ms
      y: 1D array signal (same length as t_ms)
      t0_ms,t1_ms: window
      min_peak_prom: if set, require peak height >= (median + min_peak_prom)
                    (simple robust threshold, not true prominence)
      min_peak_dist_ms: if set, enforce minimum time separation between peaks

    Returns:
      peak_times_ms: 1D array of detected peak times (ms)
      peak_values: 1D array of peak amplitudes
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    # Window
    m = (t_ms >= t0_ms) & (t_ms <= t1_ms)
    t = t_ms[m]
    s = y[m]
    if len(s) < 3:
        return np.array([]), np.array([])

    # Local maxima: s[i-1] < s[i] >= s[i+1]
    mid = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    idx = np.where(mid)[0] + 1

    if idx.size == 0:
        return np.array([]), np.array([])

    # Optional robust amplitude threshold
    if min_peak_prom is not None:
        thr = np.median(s) + float(min_peak_prom)
        idx = idx[s[idx] >= thr]
        if idx.size == 0:
            return np.array([]), np.array([])

    # Optional minimum peak distance
    if min_peak_dist_ms is not None and idx.size > 1:
        min_dist = float(min_peak_dist_ms)
        kept = [idx[0]]
        last_t = t[idx[0]]
        for k in idx[1:]:
            if (t[k] - last_t) >= min_dist:
                kept.append(k)
                last_t = t[k]
            else:
                # if too close, keep the larger of the two
                if s[k] > s[kept[-1]]:
                    kept[-1] = k
                    last_t = t[k]
        idx = np.array(kept, dtype=int)

    return t[idx], s[idx]

def burst_times_from_spikes(spike_times_ms, bin_ms=10.0, min_burst_dist_ms=136.0, height_frac=0.3):
    if spike_times_ms.size == 0:
        return np.array([])

    t0 = spike_times_ms.min()
    t1 = spike_times_ms.max()

    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    hist, edges = np.histogram(spike_times_ms, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # distance is in samples (bins)
    dist_bins = max(1, int(round(min_burst_dist_ms / bin_ms)))

    peaks, props = find_peaks(
        hist,
        height=np.max(hist) * height_frac,
        distance=dist_bins
    )
    return centers[peaks]



# phase calculations 


def calculate_peak_to_peak_phase_diag(
    sig1, sig2,
    bin_ms,
    min_peak_height=0.15,
    min_dist_ms=150.0,
    prominence=0.1,
    max_pair_ms=None,   # optional gating
):
    diag = {}

    min_dist_bins = int(round(min_dist_ms / bin_ms))
    p1, props1 = find_peaks(sig1, height=min_peak_height, distance=min_dist_bins, prominence=prominence)
    p2, props2 = find_peaks(sig2, height=min_peak_height, distance=min_dist_bins, prominence=prominence)

    diag["bin_ms"] = bin_ms
    diag["min_dist_bins"] = min_dist_bins
    diag["n_peaks_1"] = int(len(p1))
    diag["n_peaks_2"] = int(len(p2))
    diag["peaks_1_bins"] = p1
    diag["peaks_2_bins"] = p2

    if len(p1) < 2 or len(p2) < 2:
        diag["reason"] = "too_few_peaks"
        return np.nan, np.nan, np.nan, np.nan, np.nan, diag

    # periods in ms
    per1 = np.diff(p1) * bin_ms
    per2 = np.diff(p2) * bin_ms
    period1 = float(np.nanmean(per1))
    period2 = float(np.nanmean(per2))
    T = float(np.nanmean([period1, period2]))

    diag["period1_ms"] = period1
    diag["period2_ms"] = period2
    diag["T_ms"] = T

    if not np.isfinite(T) or T <= 0:
        diag["reason"] = "bad_period"
        return np.nan, np.nan, np.nan, np.nan, np.nan, diag

    freq1 = 1000.0 / period1 if period1 > 0 else np.nan
    freq2 = 1000.0 / period2 if period2 > 0 else np.nan

    # Pair peaks: nearest, but optionally gate by max_pair_ms
    phase_deg = []
    pairs = []
    for pk1 in p1:
        j = int(np.argmin(np.abs(p2 - pk1)))
        pk2 = int(p2[j])
        dt_ms = (pk2 - pk1) * bin_ms

        if max_pair_ms is not None and abs(dt_ms) > max_pair_ms:
            continue

        ph = (dt_ms / T) * 360.0
        ph = ph % 360.0
        phase_deg.append(ph)
        pairs.append((pk1, pk2, dt_ms, ph))

    diag["n_pairs"] = int(len(phase_deg))
    diag["pairs"] = pairs

    if len(phase_deg) < 2:
        diag["reason"] = "too_few_pairs"
        return np.nan, np.nan, np.nan, freq1, freq2, diag

    phase_deg = np.asarray(phase_deg, dtype=float)

    # Circular mean is better than linear mean for angles
    ang = np.deg2rad(phase_deg)
    circ_mean = (np.rad2deg(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) + 360) % 360
    circ_var = 1.0 - np.sqrt(np.mean(np.cos(ang))**2 + np.mean(np.sin(ang))**2)

    diag["phase_deg"] = phase_deg
    diag["circ_mean_deg"] = float(circ_mean)
    diag["circ_var"] = float(circ_var)

    # optional: linear stats too (sometimes useful)
    lin_mean = float(np.nanmean(phase_deg))
    lin_var  = float(np.nanvar(phase_deg))
    diag["lin_mean_deg"] = lin_mean
    diag["lin_var"] = lin_var

    return circ_mean, circ_var, np.nan, freq1, freq2, diag


def plateau_debug_report(label, spikes_L, spikes_R, t0, t1, bin_ms, L_ts, R_ts, diag, max_spikes_print=25):
    # spike counts and a small sample of raw spike times
    spikes_L = np.asarray(spikes_L, dtype=float)
    spikes_R = np.asarray(spikes_R, dtype=float)

    wL = spikes_L[(spikes_L >= t0) & (spikes_L < t1)]
    wR = spikes_R[(spikes_R >= t0) & (spikes_R < t1)]

    print(f"\n[PLATEAU DEBUG] {label}  t={t0:.1f}->{t1:.1f} ms  bin_ms={bin_ms}")
    print(f"  L spikes: {wL.size}  R spikes: {wR.size}")
    print(f"  L spike sample: {np.array2string(wL[:max_spikes_print], precision=1, separator=', ')}")
    print(f"  R spike sample: {np.array2string(wR[:max_spikes_print], precision=1, separator=', ')}")

    print(f"  L_ts stats: min={np.min(L_ts):.3g} max={np.max(L_ts):.3g} mean={np.mean(L_ts):.3g}")
    print(f"  R_ts stats: min={np.min(R_ts):.3g} max={np.max(R_ts):.3g} mean={np.mean(R_ts):.3g}")

    # peak + period diagnostics
    print(f"  peaks: n1={diag.get('n_peaks_1')} n2={diag.get('n_peaks_2')} pairs={diag.get('n_pairs')}")
    if "period1_ms" in diag:
        print(f"  periods: p1={diag['period1_ms']:.2f} ms  p2={diag['period2_ms']:.2f} ms  T={diag['T_ms']:.2f} ms")
    if "circ_mean_deg" in diag:
        print(f"  phase: circ_mean={diag['circ_mean_deg']:.2f} deg  circ_var={diag['circ_var']:.3f}  lin_mean={diag['lin_mean_deg']:.2f}")
    if "reason" in diag:
        print(f"  reason: {diag['reason']}")



def circ_mean_deg(phases_deg):
    ang = np.deg2rad(phases_deg)
    return (np.rad2deg(np.arctan2(np.nanmean(np.sin(ang)), np.nanmean(np.cos(ang)))) + 360) % 360

# optional: normalize per plateau (helps with peak height threshold consistency)
def minmax_safe(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    rng = hi - lo
    if (not np.isfinite(rng)) or (rng < eps):
        return None
    return (x - lo) / (rng + 1e-12)

def binned_rate_from_spike_times(
    spike_times_ms,
    t0_ms,
    t1_ms,
    bin_ms=5.0,
):
    """
    Convert spike times into a binned population firing-rate time series.

    Parameters
    ----------
    spike_times_ms : array-like
        Spike times in milliseconds (e.g. from window_rates_from_spike_detector).
    t0_ms, t1_ms : float
        Start and end of the time window (ms).
    bin_ms : float
        Bin width in milliseconds.

    Returns
    -------
    rate_hz : np.ndarray
        1D array of population firing rate (Hz) per bin.
    """
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)

    # Define bin edges
    edges = np.arange(t0_ms, t1_ms + bin_ms, bin_ms, dtype=float)

    if spike_times_ms.size == 0 or edges.size < 2:
        return np.zeros(max(0, len(edges) - 1), dtype=float)

    # Histogram spike counts
    counts, _ = np.histogram(spike_times_ms, bins=edges)

    # Convert counts → rate (Hz)
    rate_hz = counts / (bin_ms / 1000.0)

    return rate_hz


def debug_synapse_weights(tag, conns, sample_n=10):
    w = np.asarray(nest.GetStatus(conns, "weight"), dtype=float)

    # robust stats
    n = w.size
    n_neg = int(np.sum(w < 0))
    n_pos = int(np.sum(w > 0))
    n_zero = int(np.sum(w == 0))
    n_nan = int(np.sum(~np.isfinite(w)))

    print(f"\n[WEIGHT-DEBUG] {tag}")
    print(f"  N={n}  neg={n_neg}  pos={n_pos}  zero={n_zero}  nan/inf={n_nan}")
    if n:
        print(f"  min={np.nanmin(w):.6g}  max={np.nanmax(w):.6g}  mean={np.nanmean(w):.6g}  std={np.nanstd(w):.6g}")

        # show a few raw values (first and random)
        head = w[:min(sample_n, n)]
        print(f"  head[{head.size}]={np.array2string(head, precision=6, separator=', ')}")

        if n > sample_n:
            idx = np.random.choice(n, size=min(sample_n, n), replace=False)
            samp = w[idx]
            print(f"  rand[{samp.size}]={np.array2string(samp, precision=6, separator=', ')}")

    # also print connection metadata from one synapse (if available)
    try:
        st0 = nest.GetStatus(conns)[0]
        syn_model = st0.get("synapse_model", None)
        receptor_type = st0.get("receptor_type", None)
        receptor = st0.get("receptor", None)
        print(f"  meta: synapse_model={syn_model} receptor_type={receptor_type} receptor={receptor}")
    except Exception as e:
        print(f"  meta: (could not read per-conn metadata) {e}")


def print_last_conn_stats(conn, key, n_head=8):
    """
    Prints a quick readout for the most recently created SynapseCollection for `key`.
    Assumes ConnectNetwork stores synapses as: conn.synapses[key] = [SynapseCollection, ...]
    """
    blocks = conn.synapses.get(key, [])
    if not blocks:
        print(f"[CONN-READOUT] {key}: no SynapseCollections recorded")
        return

    sc = blocks[-1]  # the one just created
    n = len(sc)
    if n == 0:
        print(f"[CONN-READOUT] {key}: last SynapseCollection is empty")
        return

    w = np.asarray(nest.GetStatus(sc, "weight"), dtype=float)
    neg = int(np.sum(w < 0))
    pos = int(np.sum(w > 0))
    zer = int(np.sum(w == 0))

    print(f"[CONN-READOUT] {key} (LAST BLOCK)")
    print(f"  n={n}  neg={neg}  pos={pos}  zero={zer}")
    print(f"  mean={float(w.mean()):.6f}  std={float(w.std()):.6f}  min={float(w.min()):.6f}  max={float(w.max()):.6f}")
    print(f"  head[{min(n_head,n)}]={np.round(w[:min(n_head,n)], 6).tolist()}")

    # optional: confirm endpoints
    sample_n = min(5, n)
    st = nest.GetStatus(sc[:sample_n], ["source", "target", "weight"])
    print(f"  sample(source,target,weight)={st}")