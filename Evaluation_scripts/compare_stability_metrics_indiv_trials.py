#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import glob
from scipy.signal import find_peaks
np.set_printoptions(legacy='1.25',suppress=True)

plt.rcParams.update({'font.size': 20})
time_resolution = 0.1
data_by_folder = {}  # To store results for each folder

def analyze_output(input_1, input_2, pop_type, y_line_bd, y_line_phase, min_dist):
    try:
        pop_data1 = np.loadtxt(input_1, delimiter=',', dtype=float)  # Flexor
        pop_data2 = np.loadtxt(input_2, delimiter=',', dtype=float)  # Extensor

        # Normalised for threshold detection only
        pop_data1_norm = (pop_data1 - np.min(pop_data1)) / (np.max(pop_data1) - np.min(pop_data1))
        pop_data2_norm = (pop_data2 - np.min(pop_data2)) / (np.max(pop_data2) - np.min(pop_data2))

        # Burst durations
        _, _, burst_duration_flx, _, _ = calculate_burst_duration(pop_data1_norm, y_line_bd)
        _, _, burst_duration_ext, _, _ = calculate_burst_duration(pop_data2_norm, y_line_bd)

        # Frequencies
        freq_flx = calculate_freq(calculate_burst_duration(pop_data1_norm, y_line_bd)[0])
        freq_ext = calculate_freq(calculate_burst_duration(pop_data2_norm, y_line_bd)[0])

        # Phase
        phase_deg, _, _, _, _, _, _, _ = calculate_peak_to_peak_phase(
            pop_data1_norm, pop_data2_norm, y_line_phase, min_dist
        )
        phase_deg = 360 - phase_deg if phase_deg > 180 else phase_deg

        return [
            round(burst_duration_flx, 2),
            round(burst_duration_ext, 2),
            round(freq_flx, 2),
            round(freq_ext, 2),
            round(phase_deg, 2)
        ]

    except Exception as e:
        print(f"Skipping trial due to error: {e}")
        return [np.nan] * 5


def find_csv_files(folder):
    """
    Search for all output_mnp1.csv and output_mnp2.csv files within the folder and its subfolders.
    
    :param folder: The root folder to search.
    :return: List of tuples containing pairs of file paths (output_mnp1.csv, output_mnp2.csv).
    """
    # Find all output_mnp1.csv and output_mnp2.csv files
    mnp1_files = glob.glob(os.path.join(folder, '**', 'output_mnp1.csv'), recursive=True)
    mnp2_files = glob.glob(os.path.join(folder, '**', 'output_mnp2.csv'), recursive=True)
    
    # Match the mnp1 and mnp2 files by their folder structure
    file_pairs = []
    for mnp1 in mnp1_files:
        folder1 = os.path.dirname(mnp1)
        relative_path = os.path.relpath(folder1, folder)
        subfolder_name = relative_path.split(os.sep)[0]  # Get the first subfolder directly below the provided folder
        #subfolder_name = os.path.basename(folder1)  # Extract subfolder name
        corresponding_mnp2 = os.path.join(folder1, 'output_mnp2.csv')
        if corresponding_mnp2 in mnp2_files:
            file_pairs.append((mnp1, corresponding_mnp2, subfolder_name))
    
    return file_pairs

def calculate_burst_duration(array, value):
    upward_count = 0
    downward_count = 0
    upward_indices = []  # Store indices of upward crossings
    downward_indices = []  # Store indices of downward crossings

    crossing = False

    for index, item in enumerate(array):
        if item >= value and not crossing:
            upward_count += 1
            upward_indices.append(index)
            crossing = True
        elif item < value and crossing:
            downward_count += 1
            downward_indices.append(index)
            crossing = False
    min_length = min(len(downward_indices), len(upward_indices))
    downward_indices = downward_indices[1:min_length]
    upward_indices = upward_indices[1:min_length]
    burst_duration=(np.subtract(downward_indices,upward_indices))*time_resolution
    bd_variance=np.var(burst_duration)
    coeff_bd_variance=(np.std(burst_duration)/np.mean(burst_duration))
    burst_duration=np.mean(burst_duration)
    return upward_indices,downward_indices,round(burst_duration,2),round(bd_variance,2),coeff_bd_variance

def calculate_peak_to_peak_phase(spike_bins1, spike_bins2, min_peak_height, min_dist):
    pop1_peaks = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)[0]
    pop2_peaks = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)[0]

    alternating_peaks1 = []
    alternating_peaks2 = []

    i, j = 0, 0
    last_pop = None

    while i < len(pop1_peaks) and j < len(pop2_peaks):
        if last_pop is None or last_pop == 2:
            if pop1_peaks[i] < pop2_peaks[j]:
                alternating_peaks1.append(pop1_peaks[i])
                last_pop = 1
                i += 1
            else:
                #alternating_peaks2.append(pop2_peaks[j])
                #last_pop = 2
                j += 1
        elif last_pop == 1 and (pop2_peaks[j] < pop1_peaks[i]):
            alternating_peaks2.append(pop2_peaks[j])
            last_pop = 2
            j += 1
        else:
            #alternating_peaks1.append(pop1_peaks[i])
            #last_pop = 1
            i += 1

    # Truncate to the shortest length
    min_length = min(len(alternating_peaks1), len(alternating_peaks2))
    alternating_peaks1 = alternating_peaks1[:min_length]
    alternating_peaks2 = alternating_peaks2[:min_length]

    # Calculate time differences
    time_diff = np.subtract(alternating_peaks2, alternating_peaks1)
    
    period1 = np.mean(np.diff(alternating_peaks1)) * time_resolution
    period2 = np.mean(np.diff(alternating_peaks2)) * time_resolution

    # Calculate frequency for period1 and period2
    freq1 = 1000 / period1
    freq2 = 1000 / period2

    # Calculate the standard deviation of periods
    std_period1 = np.std(np.diff(alternating_peaks1)) * time_resolution
    std_period2 = np.std(np.diff(alternating_peaks2)) * time_resolution

    # Coefficient of Variation (CV) for periods
    cv_freq1 = std_period1 / period1
    cv_freq2 = std_period2 / period2
    
    avg_period = (period1 + period2) / 2
    avg_freq = 1000 / avg_period

    phase = (avg_period - (time_diff * time_resolution)) / avg_period
    phase_in_deg = phase * 360
    phase_variance = np.var(phase_in_deg)
    coeff_phase_var = (np.std(phase_in_deg) / np.mean(phase_in_deg)) 
    avg_phase_in_deg = np.mean(phase * 360)
    
    # Normalize the phase to the range [0, 360)
    avg_phase_in_deg = (avg_phase_in_deg + 360) % 360

    return round(avg_phase_in_deg, 2), round(phase_variance, 2), coeff_phase_var, round(avg_freq, 2), alternating_peaks1, alternating_peaks2, cv_freq1, cv_freq2

def calculate_amplitude_cv(spike_bins1, spike_bins2, min_peak_height, min_dist):
    # Extract peaks and their properties (including heights)
    pop1_peaks, pop1_properties = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)
    pop2_peaks, pop2_properties = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)

    # Extract peak heights
    pop1_heights = pop1_properties['peak_heights']
    pop2_heights = pop2_properties['peak_heights']

    # Calculate the variance and coefficient of variation for amplitude
    amp1_variance = np.var(pop1_heights)
    amp2_variance = np.var(pop2_heights)
    coeff_amp1_var = (np.std(pop1_heights) / np.mean(pop1_heights))
    coeff_amp2_var = (np.std(pop2_heights) / np.mean(pop2_heights))

    #print('Amplitude CV (1,2)', coeff_amp1_var, coeff_amp2_var)
    
    return coeff_amp1_var, coeff_amp2_var

def calculate_freq(arr):
    period = np.mean(np.diff(arr)) * time_resolution
    freq = 1000 / period
    
    return round(freq,2)

def find_zero_overlap(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length")

    zero_durations = []  # Store durations of zero intervals
    duration = 0  # Initialize duration

    for i in range(len(arr1)):
        if arr1[i] == 0 and arr2[i] == 0:
            duration += 1  # Increment the duration
        elif duration > 0:
            zero_durations.append(duration)  # Record the duration
            duration = 0
    
    sum_zero_durations = sum(zero_durations)*time_resolution        
    'Time suppression intervals: ',zero_durations
    return sum_zero_durations

def plot_absolute_comparisons(data_by_folder):
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    scatter_color = 'red'

    #  Burst duration (Flx + Ext)
    burst_data = []
    xticks = []
    for folder, data in data_by_folder.items():
        burst_flx = data[:, 3]   # burst duration Flx (absolute)
        burst_ext = data[:, 4]   # burst duration Ext (absolute)
        burst_data.extend([burst_flx, burst_ext])
        xticks.extend([f"{folder.split('/')[-1]} (Flx)", f"{folder.split('/')[-1]} (Ext)"])
    axs[0].boxplot(burst_data, patch_artist=True)
    for j, values in enumerate(burst_data):
        axs[0].scatter(np.ones_like(values) * (j + 1), values, color=scatter_color, alpha=0.6)
    axs[0].set_title("Burst Duration (Absolute, ms)")
    axs[0].set_xticks(range(1, len(xticks)+1))
    axs[0].set_xticklabels(xticks, rotation=45, ha='right')

    # 2 Frequency (Flx + Ext)
    freq_data = []
    xticks = []
    for folder, data in data_by_folder.items():
        freq_flx = data[:, 5]   # frequency Flx (absolute)
        freq_ext = data[:, 6]   # frequency Ext (absolute)
        freq_data.extend([freq_flx, freq_ext])
        xticks.extend([f"{folder.split('/')[-1]} (Flx)", f"{folder.split('/')[-1]} (Ext)"])
    axs[1].boxplot(freq_data, patch_artist=True)
    for j, values in enumerate(freq_data):
        axs[1].scatter(np.ones_like(values) * (j + 1), values, color=scatter_color, alpha=0.6)
    axs[1].set_title("Frequency (Absolute, Hz)")
    axs[1].set_xticks(range(1, len(xticks)+1))
    axs[1].set_xticklabels(xticks, rotation=45, ha='right')

    # 3 Phase
    phase_data = []
    xticks = []
    for folder, data in data_by_folder.items():
        phase = data[:, 7]  # phase (absolute)
        phase_data.append(phase)
        xticks.append(folder.split('/')[-1])
    axs[2].boxplot(phase_data, patch_artist=True)
    for j, values in enumerate(phase_data):
        axs[2].scatter(np.ones_like(values) * (j + 1), values, color=scatter_color, alpha=0.6)
    axs[2].set_title("Phase (Absolute, Degrees)")
    axs[2].set_xticks(range(1, len(xticks)+1))
    axs[2].set_xticklabels(xticks, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def main():
    mnp_bd_y_line = 0.4
    mnp_phase_y_line = 0.4
    min_dist_phase_calc = 1000

    # Get folder names from command line arguments


    folder_containing_data = sys.argv[1]

    # Allow the script to work even if only one argument is passed
    if len(sys.argv) > 2:
        compare_trials = sys.argv[2]
    else:
        compare_trials = None

    if compare_trials =='D1':
        folders = [
            folder_containing_data+'/P0_D1',
            folder_containing_data+'/P45_D1',
            folder_containing_data+'/P63_D1',
            folder_containing_data+'/P112_D1'
        ]
    elif compare_trials == 'D2':
        folders = [
            folder_containing_data+'/P0_D2',
            folder_containing_data+'/P45_D2',
            folder_containing_data+'/P63_D2',
            folder_containing_data+'/P112_D2'
        ]
    elif compare_trials == 'P0':
        folders = [
            folder_containing_data+'/P0_D1',
            folder_containing_data+'/P0_D2'
        ]
    elif compare_trials == 'P45':
        folders = [
            folder_containing_data+'/P45_D1',
            folder_containing_data+'/P45_D2'
        ] 
    elif compare_trials == 'P63':
        folders = [
            folder_containing_data+'/P63_D1',
            folder_containing_data+'/P63_D2'
        ] 
    elif compare_trials == 'P112':
        folders = [
            folder_containing_data+'/P112_D1',
            folder_containing_data+'/P112_D2'
        ] 
    
    # If only one dataset is provided, treat it as a single-timepoint analysis
    if compare_trials is None:
        folders = [folder_containing_data]



    global data_by_folder  # To store data for each folder
    
    for folder in folders:
        file_pairs = find_csv_files(folder)
        if not file_pairs:
            print(f"No matching output_mnp1.csv and output_mnp2.csv files found in {folder}.")
            continue

        data_array = []
        # Process each file pair in the folder
        for mnp1_input, mnp2_input, subfolder_name in file_pairs:
            data_row = analyze_output(mnp1_input, mnp2_input, 'MNP', mnp_bd_y_line, mnp_phase_y_line, min_dist_phase_calc)
            
            # Extract only the seed number before any underscores
            seed = subfolder_name.split("_")[0]
            data_array.append(np.insert(data_row, 0, int(seed)))


        
        data_by_folder[folder] = np.array(data_array, dtype=object)

    # Plot comparison results for cv_freq across folders
    plot_absolute_comparisons(data_by_folder)

if __name__ == '__main__':
    main()
