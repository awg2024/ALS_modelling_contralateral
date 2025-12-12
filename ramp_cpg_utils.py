
#!/usr/bin/env python

import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import time
import start_simulation as ss
import pickle, yaml
import pandas as pd
import warnings
from scipy.signal import find_peaks,correlate
from scipy.fft import fft, fftfreq
import set_network_params as netparams
from phase_ordering import order_by_phase
from pca import run_PCA
from connect_populations import ConnectNetwork
import population_functions as popfunc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _flatten_scalars(x):
    """
    Recursively flatten nested lists/tuples/np.ndarrays into a flat Python list
    of scalar elements (no arrays, no lists).
    """
    out = []
    if isinstance(x, (list, tuple, np.ndarray)):
        for y in x:
            out.extend(_flatten_scalars(y))
    else:
        out.append(x)
    return out



def ensure_grouped_spikes(senders, times, pop_size, neuron_ids):
    """
    Returns list-of-lists spike times (len == pop_size).
    Robust to all NEST spike formats (flat, grouped, single-block, ragged object arrays).
    """

    # Trust "already grouped" ONLY if length matches pop_size
    if (
        isinstance(times, list)
        and len(times) == pop_size
        and all(isinstance(t, (list, np.ndarray)) for t in times)
    ):
        return [list(_flatten_scalars(t)) for t in times]

    flat_senders = _flatten_scalars(senders)
    flat_times   = _flatten_scalars(times)

    # Convert to pure python scalars
    flat_senders = [int(s) for s in flat_senders]
    flat_times   = [float(t) for t in flat_times]

    # Pair safely (in case lengths mismatch)
    n = min(len(flat_senders), len(flat_times))
    flat_senders = flat_senders[:n]
    flat_times   = flat_times[:n]

    return group_spikes_by_neuron(flat_senders, flat_times, pop_size, neuron_ids)



def group_spikes_by_neuron(senders, times, pop_size, neuron_ids):
    """
    Convert flat spike lists from NEST spike_detector into a list-of-lists
    indexed per neuron (0..N-1).

    neuron_ids can be:
      - a NodeCollection
      - a list of gids
      - a list containing NodeCollections and/or gids

    We flatten everything down to a simple Python list of ints.
    """

    # --- 1) Flatten neuron_ids into a flat list of GIDs (ints) ---
    flat_ids = []

    def _flatten_ids(x):
        # NEST NodeCollection (any size)
        if isinstance(x, nest.NodeCollection):
            for gid in x.tolist():
                flat_ids.append(int(gid))

        # Python container
        elif isinstance(x, (list, tuple)):
            for y in x:
                _flatten_ids(y)

        # Scalar GID
        else:
            flat_ids.append(int(x))

    # 🔑 THIS LINE WAS MISSING
    _flatten_ids(neuron_ids)

    # --- Safety check ---
    n_neurons = len(flat_ids)
    if pop_size != n_neurons:
        print(
            f"[WARN] pop_size={pop_size} but len(neuron_ids)={n_neurons}. "
            f"Using n_neurons={n_neurons} for grouping."
        )
        pop_size = n_neurons

    # --- 2) Prepare output list-of-lists ---
    grouped = [[] for _ in range(pop_size)]

    # Map gid -> row index
    gid_to_index = {gid: i for i, gid in enumerate(flat_ids)}

    # --- 3) Fill spike times per neuron ---
    for s, t in zip(senders, times):
        s = int(s)   # 🔑 force hashable scalar
        idx = gid_to_index.get(s)

        if idx is not None:
            grouped[idx].append(t)

    return grouped



# ==============================================================
# 3. Convolution of all spike trains (PER-NEURON VERSION)
# ==============================================================

def convolve_per_neuron(pop_size, spike_times, kernel_sigma=20.0):
    """
    Convert list-of-lists spike data into a matrix:
        shape = (N_neurons × T)
    using Gaussian convolution.
    """

    
    spikes = spike_times

    # --------------------------------------------------
    # 1. Find maximum spike time across population
    # --------------------------------------------------
    max_t = 0
    for n in range(pop_size):
        if len(spikes[n]) > 0:
            max_t = max(max_t, max(spikes[n]))

    # --------------------------------------------------
    # 2. Build Gaussian kernel
    # --------------------------------------------------
    sigma = kernel_sigma
    kernel_radius = int(4 * sigma)
    kernel_t = np.arange(-kernel_radius, kernel_radius)
    kernel = np.exp(-(kernel_t ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()


    # Ensure a minimum time axis even if population is silent
    min_T = int(8 * kernel_sigma)  # enough to hold kernel safely

    if max_t == 0:
        max_t = min_T
    else:
        max_t = max(int(max_t) + 5, min_T)

    # Time base
    t = np.arange(0, max_t, 1)


    # 4. Convolution
    # --------------------------------------------------
    conv_matrix = np.zeros((pop_size, len(t)))

    for n in range(pop_size):
        spike_train = np.zeros(len(t))
        for sp in spikes[n]:
            sp_i = int(sp)
            if sp_i < len(t):
                spike_train[sp_i] += 1

        conv_matrix[n, :] = np.convolve(spike_train, kernel, mode="same")

    return conv_matrix, t





    # ==============================================================
    #  7. HEATMAPS
    # ==============================================================

def heatmap(name, data, scale, colorpath):
    plt.figure(figsize=(14, 6))
    plt.imshow(data * scale, aspect='auto', cmap="plasma")
    plt.colorbar(label="Hz")
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.savefig(colorpath, dpi=300)
    plt.close()




def cpg_utils(
        nn, popfunc,
        L_rg1, L_rg2, R_rg1, R_rg2,
        L_V2a, R_V2a,
        L_V0V, R_V0V,
        L_V0D, R_V0D,
        L_mnp1, L_mnp2, R_mnp1, R_mnp2,
        label
    ):

    sp_v0v_left = None
    sp_v0v_right = None
    v0v_left_conv = None
    v0v_right_conv = None




    """
    STREAMLINED VERSION:
    - Extracts spikes
    - Computes ISF
    - Convolves spikes
    - Applies ISF scaling
    - Produces:
        RG–V0d–MNP plot
        RG–V2a–V0v–MNP plot
        Heatmaps for V0 populations
    """

    # ==============================================================
    # 🧠 1. Extract spike data
    # ==============================================================

    #Check integrity of V0v and V0d creation. 
    print("integrity of v0v and v0d during cpg_utils...")
    print("L_V0V.v0v_all type:", type(L_V0V.v0v_all), "len:", len(L_V0V.v0v_all))
    print("R_V0V.v0v_all type:", type(R_V0V.v0v_all), "len:", len(R_V0V.v0v_all))
    print("L_V0D.v0d_all type:", type(L_V0D.v0d_all), "len:", len(L_V0D.v0d_all))
    print("R_V0D.v0d_all type:", type(R_V0D.v0d_all), "len:", len(R_V0D.v0d_all))
    time.sleep(10)




    # RG populations (bursting)
    _, sp_rg1 = popfunc.read_spike_data(L_rg1.spike_detector_rg_exc_bursting)
    _, sp_rg2 = popfunc.read_spike_data(L_rg2.spike_detector_rg_exc_bursting)
    _, sp_rg1_contra = popfunc.read_spike_data(R_rg1.spike_detector_rg_exc_bursting)
    _, sp_rg2_contra = popfunc.read_spike_data(R_rg2.spike_detector_rg_exc_bursting)

    # V2a (tonic excitatory interneurons)
    _, sp_v2a_left_raw  = popfunc.read_spike_data(L_V2a.spike_detector_exc_inter_tonic)
    _, sp_v2a_right_raw = popfunc.read_spike_data(R_V2a.spike_detector_exc_inter_tonic)

    sp_v2a_left  = sp_v2a_left_raw[0]
    sp_v2a_right = sp_v2a_right_raw[0]
    
    print("[DEBUG] V2a left neurons:", len(sp_v2a_left),
      "type[0]:", type(sp_v2a_left[0]),
      "len[0]:", len(sp_v2a_left[0]) if len(sp_v2a_left) else None)



    if nn.ramp_experimental_v0v_weights:

        # V0v populations
        send_v0v_L, time_v0v_L = popfunc.read_spike_data(L_V0V.spike_detector)
        send_v0v_R, time_v0v_R = popfunc.read_spike_data(R_V0V.spike_detector)

        # Debugging steps
        # print(f"[DEBUG] send_v0v_L : {send_v0v_L}")
        # print(f"[DEBUG] time_v0v_L : {time_v0v_L}")
        # print(f"[DEBUG] v0v pop size : {nn.v0v_pop_size}")
        # print(f"[DEBUG] L_V0V.v0v_all: {L_V0V.v0v_all}")
        # time.sleep(10)

        sp_v0v_left  = time_v0v_L[0]
        sp_v0v_right = time_v0v_R[0]


        # sp_v0v_left  = group_spikes_by_neuron(send_v0v_L,  time_v0v_L,
        #                                     nn.v0v_pop_size, L_V0V.v0v_all)

        # sp_v0v_right = group_spikes_by_neuron(send_v0v_R,  time_v0v_R,
        #                                     nn.v0v_pop_size, R_V0V.v0v_all)

        print("[DEBUG] v0v_left: n_neurons =", len(sp_v0v_left),
        "example spikes lens:", [len(sp_v0v_left[i]) for i in range(min(5, len(sp_v0v_left)))])

    if nn.ramp_experimental_v0d_weights:


        # V0d Population
        send_v0d_L, time_v0d_L = popfunc.read_spike_data(L_V0D.spike_detector)
        send_v0d_R, time_v0d_R = popfunc.read_spike_data(R_V0D.spike_detector)


            # Debugging steps
        # print(f"[DEBUG] send_v0d_L : {send_v0d_L}")
        # print(f"[DEBUG] time_v0d_L : {time_v0d_L}")
        # print(f"[DEBUG] v0v pop size : {nn.v0d_pop_size}")
        # print(f"[DEBUG] L_V0V.v0v_all: {L_V0D.v0d_all}")
        # time.sleep(10)

        # V0d spikes are already grouped per neuron
        # Unwrap single outer list returned by read_spike_data
        
        sp_v0d_left = ensure_grouped_spikes(
            send_v0d_L,
            time_v0d_L,
            nn.v0d_pop_size,
            L_V0D.v0d_all
        )

        sp_v0d_right = ensure_grouped_spikes(
            send_v0d_R,
            time_v0d_R,
            nn.v0d_pop_size,
            R_V0D.v0d_all
        )

        print("[DEBUG] v0d_left: n_neurons =", len(sp_v0d_left),
        "example spikes lens:", [len(sp_v0d_left[i]) for i in range(min(5, len(sp_v0d_left)))])


    # unwrap one level
    sp_rg1 = sp_rg1[0]
    sp_rg2 = sp_rg2[0]
    sp_rg1_contra = sp_rg1_contra[0]
    sp_rg2_contra = sp_rg2_contra[0]


    # ==============================================================
    # 🧠 2. Compute instantaneous frequency (ISF)
    # ==============================================================

    def isf_and_scale(pop_size, spk, convolved):
        """Compute ISF scale factor for population."""
        freq, _ = popfunc.calculate_interspike_frequency(pop_size, spk)
        isf_max = np.nanmax([np.nanmean(f) for f in freq]) if len(freq) else 1
        conv_max = np.nanmax(convolved) if np.nanmax(convolved) > 0 else 1
        scale = isf_max / conv_max
        return scale

    # ==============================================================
    # 🧠 3. Convolution of all spike trains
    # ==============================================================

    if nn.ramp_experimental_v0d_weights:

        # RG
        v0d_left_conv, t = convolve_per_neuron(nn.v0d_pop_size, sp_v0d_left)
        v0d_right_conv, _ = convolve_per_neuron(nn.v0d_pop_size, sp_v0d_right)

        v0d_left_scale  = isf_and_scale(nn.v0d_pop_size, sp_v0d_left, v0d_left_conv)
        v0d_right_scale = isf_and_scale(nn.v0d_pop_size, sp_v0d_right, v0d_right_conv)

    rg1_conv, t = convolve_per_neuron(nn.flx_exc_bursting_count, sp_rg1)
    rg2_conv, _ = convolve_per_neuron(nn.ext_exc_bursting_count, sp_rg2)
    rg1_contra_conv, _ = convolve_per_neuron(nn.flx_exc_bursting_count, sp_rg1_contra)
    rg2_contra_conv, _ = convolve_per_neuron(nn.ext_exc_bursting_count, sp_rg2_contra)

    v2a_left_conv, _ = convolve_per_neuron(nn.v2a_tonic_pop_size, sp_v2a_left)
    v2a_right_conv, _ = convolve_per_neuron(nn.v2a_tonic_pop_size, sp_v2a_right)


    if nn.ramp_experimental_v0v_weights:

        v0v_left_conv, _ = convolve_per_neuron(nn.v0v_pop_size, sp_v0v_left)
        v0v_right_conv, _ = convolve_per_neuron(nn.v0v_pop_size, sp_v0v_right)

        v0v_left_scale  = isf_and_scale(nn.v0v_pop_size, sp_v0v_left, v0v_left_conv)
        v0v_right_scale = isf_and_scale(nn.v0v_pop_size, sp_v0v_right, v0v_right_conv)

    
    # ==========================
# MNP spike extraction
# ==========================
    _, sp_mnp1_raw = popfunc.read_spike_data(L_mnp1.spike_detector)
    _, sp_mnp2_raw = popfunc.read_spike_data(L_mnp2.spike_detector)
    _, sp_mnp1_contra_raw = popfunc.read_spike_data(R_mnp1.spike_detector)
    _, sp_mnp2_contra_raw = popfunc.read_spike_data(R_mnp2.spike_detector)

    sp_mnp1 = sp_mnp1_raw[0]
    sp_mnp2 = sp_mnp2_raw[0]
    sp_mnp1_contra = sp_mnp1_contra_raw[0]
    sp_mnp2_contra = sp_mnp2_contra_raw[0]

    mnp1_conv, _ = convolve_per_neuron(nn.num_motor_neurons, sp_mnp1)
    mnp2_conv, _ = convolve_per_neuron(nn.num_motor_neurons, sp_mnp2)
    mnp1_contra_conv, _ = convolve_per_neuron(nn.num_motor_neurons, sp_mnp1_contra)
    mnp2_contra_conv, _ = convolve_per_neuron(nn.num_motor_neurons, sp_mnp2_contra)


    # ==============================================================
    # 🧠 4. ISF scaling
    # ==============================================================

    rg1_scale  = isf_and_scale(nn.flx_exc_bursting_count, sp_rg1, rg1_conv)
    rg2_scale  = isf_and_scale(nn.ext_exc_bursting_count, sp_rg2, rg2_conv)
    rg1_contra_scale = isf_and_scale(nn.flx_exc_bursting_count, sp_rg1_contra, rg1_contra_conv)
    rg2_contra_scale = isf_and_scale(nn.ext_exc_bursting_count, sp_rg2_contra, rg2_contra_conv)

    v2a_left_scale = isf_and_scale(nn.v2a_tonic_pop_size, sp_v2a_left, v2a_left_conv)
    v2a_right_scale = isf_and_scale(nn.v2a_tonic_pop_size, sp_v2a_right, v2a_right_conv)

    mnp1_scale = isf_and_scale(nn.num_motor_neurons, sp_mnp1, mnp1_conv)
    mnp2_scale = isf_and_scale(nn.num_motor_neurons, sp_mnp2, mnp2_conv)
    mnp1_contra_scale = isf_and_scale(nn.num_motor_neurons, sp_mnp1_contra, mnp1_contra_conv)
    mnp2_contra_scale = isf_and_scale(nn.num_motor_neurons, sp_mnp2_contra, mnp2_contra_conv)

    # ==============================================================
    # 📊 5. PLOTS — V0d panel
    # ==============================================================

    if nn.ramp_experimental_v0d_weights:

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1.2])

        # RG
        ax_rg_L = fig.add_subplot(gs[0, 0])
        ax_rg_R = fig.add_subplot(gs[0, 1])

        ax_rg_L.plot(t, rg1_conv * rg1_scale)
        ax_rg_L.plot(t, rg2_conv * rg2_scale)

        ax_rg_R.plot(t, rg1_contra_conv * rg1_contra_scale, '--')
        ax_rg_R.plot(t, rg2_contra_conv * rg2_contra_scale, '--')

        # V0d
        ax_v0d_L = fig.add_subplot(gs[1, 0])
        ax_v0d_R = fig.add_subplot(gs[1, 1])

        ax_v0d_L.plot(t, v0d_left_conv * v0d_left_scale)
        ax_v0d_R.plot(t, v0d_right_conv * v0d_right_scale)

        # MNP
        ax_mnp_L = fig.add_subplot(gs[2, 0])
        ax_mnp_R = fig.add_subplot(gs[2, 1])

        ax_mnp_L.plot(t, mnp1_conv * mnp1_scale)
        ax_mnp_L.plot(t, mnp2_conv * mnp2_scale)

        ax_mnp_R.plot(t, mnp1_contra_conv * mnp1_contra_scale, '--')
        ax_mnp_R.plot(t, mnp2_contra_conv * mnp2_contra_scale, '--')

        plt.savefig(f"{nn.pathFigures}/{label}_V0d_block.png", dpi=300)
        plt.close()

    # ==============================================================
    # 📊 6. PLOTS — V0v panel (with V2a)
    # ==============================================================

    if nn.ramp_experimental_v0v_weights:

        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 1, 1, 1.2])

        # RG
        ax_rg_L = fig.add_subplot(gs[0, 0])
        ax_rg_R = fig.add_subplot(gs[0, 1])

        ax_rg_L.plot(t, rg1_conv * rg1_scale)
        ax_rg_L.plot(t, rg2_conv * rg2_scale)

        ax_rg_R.plot(t, rg1_contra_conv * rg1_contra_scale, '--')
        ax_rg_R.plot(t, rg2_contra_conv * rg2_contra_scale, '--')

        # V2a
        ax_v2a_L = fig.add_subplot(gs[1, 0])
        ax_v2a_R = fig.add_subplot(gs[1, 1])

        ax_v2a_L.plot(t, v2a_left_conv * v2a_left_scale)
        ax_v2a_R.plot(t, v2a_right_conv * v2a_right_scale)

        # V0v
        ax_v0v_L = fig.add_subplot(gs[2, 0])
        ax_v0v_R = fig.add_subplot(gs[2, 1])

        ax_v0v_L.plot(t, v0v_left_conv * v0v_left_scale)
        ax_v0v_R.plot(t, v0v_right_conv * v0v_right_scale)

        # MNP
        ax_mnp_L = fig.add_subplot(gs[3, 0])
        ax_mnp_R = fig.add_subplot(gs[3, 1])

        ax_mnp_L.plot(t, mnp1_conv * mnp1_scale)
        ax_mnp_L.plot(t, mnp2_conv * mnp2_scale)

        ax_mnp_R.plot(t, mnp1_contra_conv * mnp1_contra_scale, '--')
        ax_mnp_R.plot(t, mnp2_contra_conv * mnp2_contra_scale, '--')

        plt.savefig(f"{nn.pathFigures}/{label}_V0v_block.png", dpi=300)
        plt.close()
    

    if nn.ramp_experimental_v0v_weights:
        heatmap("V0v ipsi", v0v_left_conv, v0v_left_scale,
                f"{nn.pathFigures}/{label}_heatmap_V0v_ipsi.png")
        heatmap("V0v contra", v0v_right_conv, v0v_right_scale,
                f"{nn.pathFigures}/{label}_heatmap_V0v_contra.png")


    if nn.ramp_experimental_v0d_weights:
        heatmap("V0d ipsi", v0d_left_conv, v0d_left_scale,
                f"{nn.pathFigures}/{label}_heatmap_V0d_ipsi.png")
        heatmap("V0d contra", v0d_right_conv, v0d_right_scale,
                f"{nn.pathFigures}/{label}_heatmap_V0d_contra.png")




