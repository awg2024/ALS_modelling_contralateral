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
from scipy.signal import find_peaks


nn=netparams.neural_network()

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

    spike_data = population[neuron_number]  # assuming population[neuron_number] gives array/list of spike times

    if isinstance(spike_data, (list, np.ndarray)):
        # If it's something like [array([...]), array([...]), ...]
        if len(spike_data) > 0 and isinstance(spike_data[0], (list, np.ndarray)):
            # Flatten by concatenating each segment
            segments = [np.ravel(seg) for seg in spike_data if seg is not None]
            if segments:
                spike_data = np.concatenate(segments)
            else:
                spike_data = np.array([], dtype=float)
        else:
            spike_data = np.ravel(spike_data)
    else:
        spike_data = np.array([], dtype=float)

    for t in spike_data:
        try:
            spike_time_index = int(t / nn.time_resolution)
            if 0 <= spike_time_index < n_bins:
                spike_time[spike_time_index] = 1
        except Exception:
            # Skip malformed entries
            continue

    return spike_time


'''
def calculate_interspike_frequency(neuron_count, output_spiketimes):
    frequencies = []
    times = []
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        if len(t_spikes) > 0:
            # Sort spikes by time
            sorted_indices = np.argsort(t_spikes)
            spike_times = t_spikes[sorted_indices]
            
            isi = np.diff(t_spikes)
            frequencies.append(1000.0 / isi)
            times.append(spike_times[1:])
    return frequencies, times
'''

def calculate_interspike_frequency(neuron_count, output_spiketimes):
    frequencies = []
    times = []
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        if len(t_spikes) > 1:  # Need at least two spikes for ISI
            # Sort spikes by time
            sorted_indices = np.argsort(t_spikes)
            spike_times = t_spikes[sorted_indices]
            
            isi = np.diff(spike_times)
            # Filter out NaNs
            valid_mask = ~np.isnan(isi)
            valid_isi = isi[valid_mask]
            valid_times = spike_times[1:][valid_mask]
            
            if len(valid_isi) > 0:
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
	# Initialize the spike bins array as a 2D array
	bins=np.arange(0, nn.sim_time+nn.time_resolution,nn.time_resolution)
	# Loop over each neuron
	for i in range(neuron_count):
	    t_spikes = output_spiketimes[0][i]
	    # Use numpy's histogram function to assign each spike to its corresponding time bin index
	    spikes_per_bin,bin_edges=np.histogram(t_spikes, bins)
	    # Add the spike counts to the `spike_bins_current` array
	    if i == 0:
	        spike_bins_current = spikes_per_bin
	    else:
	        spike_bins_current += spikes_per_bin
	spike_bins_current = sliding_time_window(spike_bins_current,nn.time_window) #Applies a time window to smooth the output        
	smoothed_spike_bins = gaussian_filter(spike_bins_current, nn.convstd_rate) #Applies a filter to smooth the high frequency noise
	if nn.chop_edges_amount > 0.0:
	    smoothed_spike_bins = smoothed_spike_bins[int(nn.chop_edges_amount):int(-nn.chop_edges_amount)]
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

    smoothed_spikes = smooth(binary_spikes, nn.convstd_rate)

    time_vector = np.arange(binary_spikes.shape[1]) * nn.time_resolution

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

    # Then existing metrics
    for k in (existing.keys() if isinstance(existing, dict) else []):
        if k in df.columns and k not in front:
            front.append(k)

    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    df.to_csv(out_csv_path, index=False)
    return df
