#!/usr/bin/env python

import nest
import numpy as np
import sys
import matplotlib.pyplot as plt
import start_simulation as ss
import set_network_params as netparams
import population_functions as popfunc
from ramp_circuitry import build_full_cpg_network
import os
import time
import population_functions as popfunc


def plot_v0v_block(
        t,
        rg1_convolved, rg2_convolved,
        contra_rg1_convolved, contra_rg2_convolved,
        v2a_left, v2a_contra,
        v0v_left, v0v_contra,
        mnp1_convolved, mnp2_convolved,
        contra_mnp1_convolved, contra_mnp2_convolved,
        scale_label, save_path
    ):

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 1.0, 1.0, 1.2])

    # --- Row 1 RG ---
    ax_rg_left = fig.add_subplot(gs[0, 0])
    ax_rg_right = fig.add_subplot(gs[0, 1])
    ax_rg_left.plot(t, rg1_convolved); ax_rg_left.plot(t, rg2_convolved)
    ax_rg_right.plot(t, contra_rg1_convolved, '--'); ax_rg_right.plot(t, contra_rg2_convolved, '--')

    # --- Row 2 V2a ---
    ax_v2a_left = fig.add_subplot(gs[1, 0])
    ax_v2a_right = fig.add_subplot(gs[1, 1])
    ax_v2a_left.plot(t, v2a_left); ax_v2a_right.plot(t, v2a_contra)

    # --- Row 3 V0v ---
    ax_v0v_left = fig.add_subplot(gs[2, 0])
    ax_v0v_right = fig.add_subplot(gs[2, 1])
    ax_v0v_left.plot(t, v0v_left); ax_v0v_right.plot(t, v0v_contra)

    # --- Row 4 MNP ---
    ax_mnp_left = fig.add_subplot(gs[3, 0])
    ax_mnp_right = fig.add_subplot(gs[3, 1])
    ax_mnp_left.plot(t, mnp1_convolved); ax_mnp_left.plot(t, mnp2_convolved)
    ax_mnp_right.plot(t, contra_mnp1_convolved, '--'); ax_mnp_right.plot(t, contra_mnp2_convolved, '--')

    plt.tight_layout()
    plt.savefig(f"{save_path}/V0v_block_scale_{scale_label}.png", dpi=300)
    plt.close()


def plot_v0d_block(
        t, rg1_convolved, rg2_convolved,
        contra_rg1_convolved, contra_rg2_convolved,
        v0d_convolved, v0d_contra_convolved,
        mnp1_convolved, mnp2_convolved,
        contra_mnp1_convolved, contra_mnp2_convolved,
        scale_label, save_path
    ):
    
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.2])

    # ----- RG -----
    ax_rg_left  = fig.add_subplot(gs[0, 0])
    ax_rg_right = fig.add_subplot(gs[0, 1])

    ax_rg_left.plot(t, rg1_convolved)
    ax_rg_left.plot(t, rg2_convolved)
    ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

    ax_rg_right.plot(t, contra_rg1_convolved, linestyle='--')
    ax_rg_right.plot(t, contra_rg2_convolved, linestyle='--')
    ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")

    # ----- V0d -----
    ax_v0d_left = fig.add_subplot(gs[1, 0])
    ax_v0d_right = fig.add_subplot(gs[1, 1])

    ax_v0d_left.plot(t, v0d_convolved)
    ax_v0d_left.set_title("V0d Ipsilateral")

    ax_v0d_right.plot(t, v0d_contra_convolved)
    ax_v0d_right.set_title("V0d Contralateral")

    # ----- MNP -----
    ax_mnp_left = fig.add_subplot(gs[2, 0])
    ax_mnp_right = fig.add_subplot(gs[2, 1])

    ax_mnp_left.plot(t, mnp1_convolved)
    ax_mnp_left.plot(t, mnp2_convolved)
    ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small")

    ax_mnp_right.plot(t, contra_mnp1_convolved)
    ax_mnp_right.plot(t, contra_mnp2_convolved)
    ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small")

    ax_mnp_left.set_xlabel("Time (ms)")
    ax_mnp_right.set_xlabel("Time (ms)")

    plt.tight_layout()
    plt.savefig(f"{save_path}/V0d_block_scale_{scale_label}.png", dpi=300)
    plt.close()


def flatten_spike_times(spike_times):
    flat = []
    def _flatten(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            for y in x:
                _flatten(y)
        else:
            flat.append(float(x))
    _flatten(spike_times)
    return np.array(flat) if len(flat) else np.array([])


def extract_block_spikes(spike_times, t_start, t_stop):
    flat = flatten_spike_times(spike_times)
    if flat.size == 0:
        return flat
    mask = (flat >= t_start) & (flat < t_stop)
    return flat[mask] - t_start


def spike_rate_from_spikes(spike_times, bin_ms=10, sim_time=500):
    bins = np.arange(0, sim_time + bin_ms, bin_ms)
    rate, _ = np.histogram(spike_times, bins=bins)
    return rate * (1000.0 / bin_ms)

# --------------------------------------
# INITIALISE NETWORK
# --------------------------------------
nn = netparams.neural_network()
nest.ResetKernel()
ss.nest_start()

# call in full network model.
pops = build_full_cpg_network()

L_mnp1, L_mnp2 = pops["L_mnp1"], pops["L_mnp2"]
R_mnp1, R_mnp2 = pops["R_mnp1"], pops["R_mnp2"]
L_V0D, R_V0D   = pops["L_V0D"], pops["R_V0D"]
L_V0V, R_V0V = pops["L_V0V"], pops["R_V0V"]
L_rg1, R_rg1   = pops["L_rg1"], pops["R_rg1"]
L_rg2, R_rg2 = pops["L_rg2"], pops["R_rg2"]
L_V2a, R_V2a = pops["L_exc1"], pops["R_exc1"]
L_V1a_1, R_V1a_1 = pops["L_V1a_1"], pops["R_V1a_1"]
L_V1a_2, R_V1a_2 = pops["L_V1a_2"], pops["R_V1a_2"]


# --------------------------------------
# Identify V0D→RG synapses (both directions)
# --------------------------------------
# L→R


if nn.ramp_experimental_v0d_weights:
    src_L = L_V0D.v0d_bursting
    tgt_R = R_rg1.rg_exc_bursting
else:
    src_L = L_V0D.v0d_tonic
    tgt_R = R_rg1.rg_exc_tonic

# R→L
if nn.ramp_experimental_v0d_weights:
    src_R = R_V0D.v0d_bursting
    tgt_L = L_rg1.rg_exc_bursting
else:
    src_R = R_V0D.v0d_tonic
    tgt_L = L_rg1.rg_exc_tonic

# ------------------------------------------
# V0V L → R
# ------------------------------------------
if nn.ramp_experimental_v0v_weights:
    # HIGH locomotion → V0v bursting inhibits RG_tonic
    src_L_v0v = L_V0V.v0v_bursting
    tgt_R_v0v = R_rg1.rg_exc_tonic
else:
    # LOW locomotion → V0v tonic inhibits RG_bursting
    src_L_v0v = L_V0V.v0v_tonic
    tgt_R_v0v = R_rg1.rg_exc_bursting

# ------------------------------------------
# V0V R → L
# ------------------------------------------
if nn.ramp_experimental_v0v_weights:
    # HIGH locomotion
    src_R_v0v = R_V0V.v0v_bursting
    tgt_L_v0v = L_rg1.rg_exc_tonic
else:
    # LOW locomotion
    src_R_v0v = R_V0V.v0v_tonic
    tgt_L_v0v = L_rg1.rg_exc_bursting


conns_LR = nest.GetConnections(source=src_L, target=tgt_R)
conns_RL = nest.GetConnections(source=src_R, target=tgt_L)

baseline_LR = np.mean(nest.GetStatus(conns_LR, "weight"))
baseline_RL = np.mean(nest.GetStatus(conns_RL, "weight"))

print(f"[INFO] V0D L→R baseline = {baseline_LR:.4f}")
print(f"[INFO] V0D R→L baseline = {baseline_RL:.4f}")

# --------------------------------------
# RAMP PARAMETERS
# --------------------------------------
scales = [0.0, 0.5, 1.0, 2.0, 8.0]
sim_time = 2000
bin_ms   = 10

block_spikes_L = []
block_spikes_R = []
block_spikes_V0d_L = []
block_spikes_V0d_R = []
rate_mnp1_peak, rate_mnp2_peak = [], []
weights = []

global_time = 0


# --------------------------------------
# MAIN LOOP
# --------------------------------------
for k, scale in enumerate(scales):
    new_LR = baseline_LR * scale
    new_RL = baseline_RL * scale
    nest.SetStatus(conns_LR, {"weight": new_LR})   
    nest.SetStatus(conns_RL, {"weight": new_RL})

    print(f"[BLOCK {k}] scale={scale:.2f}  (L→R={new_LR:.3f}, R→L={new_RL:.3f})")
    nest.Simulate(sim_time)

    block_start = global_time
    block_stop  = global_time + sim_time
    global_time += sim_time

    # Collect spikes
    _, sp_rg1_exc = popfunc.read_spike_data(L_rg1.spike_detector_rg_exc_bursting)
    _, sp_rg2_exc = popfunc.read_spike_data(L_rg2.spike_detector_rg_exc_bursting)
    _, sp_contra_rg1_exc = popfunc.read_spike_data(R_rg1.spike_detector_rg_exc_bursting)
    _, sp_contra_rg2_exc = popfunc.read_spike_data(R_rg2.spike_detector_rg_exc_bursting)

    _, sp_v2a_left = popfunc.read_spike_data(L_V2a.spike_detector_exc_inter_tonic)
    _, sp_v2a_contra = popfunc.read_spike_data(R_V2a.spike_detector_exc_inter_tonic)

    _, sp_v0v_left = popfunc.read_spike_data(L_V0V.spike_detector)
    _, sp_v0v_contra = popfunc.read_spike_data(R_V0V.spike_detector)

    _, sp_v0d_left = popfunc.read_spike_data(L_V0D.spike_detector)
    _, sp_v0d_contra = popfunc.read_spike_data(R_V0D.spike_detector)

    _, sp_mnp1 = popfunc.read_spike_data(L_mnp1.spike_detector_motor)
    _, sp_mnp2 = popfunc.read_spike_data(L_mnp2.spike_detector_motor)
    _, sp_contra_mnp1 = popfunc.read_spike_data(R_mnp1.spike_detector_motor)
    _, sp_contra_mnp2 = popfunc.read_spike_data(R_mnp2.spike_detector_motor)

    spL = extract_block_spikes(sp_mnp1, block_start, block_stop)
    spR = extract_block_spikes(sp_mnp2, block_start, block_stop)
    spLV0d = extract_block_spikes(sp_v0d_left, block_start, block_stop)
    spRV0d = extract_block_spikes(sp_v0d_contra, block_start, block_stop)


    block_spikes_L.append(spL.tolist())
    block_spikes_R.append(spR.tolist())
    block_spikes_V0d_L.append(spLV0d.tolist())
    block_spikes_V0d_R.append(spRV0d.tolist())

    weights.append(scale)
    rate_L = spike_rate_from_spikes(spL, bin_ms, sim_time)
    rate_R = spike_rate_from_spikes(spR, bin_ms, sim_time)
    rate_mnp1_peak.append(np.mean(rate_L) if len(rate_L) else 0)
    rate_mnp2_peak.append(np.mean(rate_R) if len(rate_R) else 0)


    print(f" [SANITY CHECK] → Mean MNP1={rate_mnp1_peak[-1]:.1f} Hz   MNP2={rate_mnp2_peak[-1]:.1f} Hz")

    rg1_convolved, t = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count, sp_rg1_exc)
    rg2_convolved, _ = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count, sp_rg2_exc)

    contra_rg1_convolved, _ = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count, sp_contra_rg1_exc)
    contra_rg2_convolved, _ = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count, sp_contra_rg2_exc)

    v2a1_convolved, _ = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size, sp_v2a_left)
    v2a_contra_convolved, _ = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size, sp_v2a_contra)

    v0v_convolved, _ = popfunc.convolve_spiking_activity(nn.v0v_pop_size, sp_v0v_left)
    v0v_contra_convolved, _ = popfunc.convolve_spiking_activity(nn.v0v_pop_size, sp_v0v_contra)

    v0d_convolved, _ = popfunc.convolve_spiking_activity(nn.v0d_pop_size, sp_v0d_left)
    v0d_contra_convolved, _ = popfunc.convolve_spiking_activity(nn.v0d_pop_size, sp_v0d_contra)

    mnp1_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, sp_mnp1)
    mnp2_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, sp_mnp2)
    contra_mnp1_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, sp_contra_mnp1)
    contra_mnp2_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, sp_contra_mnp2)

    # --- After computing convolved traces ---
    if nn.args['ramp_experimental_v0d_weights']:

        plot_v0d_block(
            t,
            rg1_convolved, rg2_convolved,
            contra_rg1_convolved, contra_rg2_convolved,
            v0d_convolved, v0d_contra_convolved,
            mnp1_convolved, mnp2_convolved,
            contra_mnp1_convolved, contra_mnp2_convolved,
            scale_label=f"{scale:.2f}",
            save_path=nn.pathFigures
        )

    if nn.args['ramp_experimental_v0v_weights']:
        
        # define variables for plotting. 


        plot_v0v_block(
            t,
            rg1_convolved, rg2_convolved,
            contra_rg1_convolved, contra_rg2_convolved,
            v2a1_convolved, v2a_contra_convolved,
            v0v_convolved, v0v_contra_convolved,
            mnp1_convolved, mnp2_convolved,
            contra_mnp1_convolved, contra_mnp2_convolved,
            scale_label=f"{scale:.2f}",
            save_path=nn.pathFigures
        )

# --------------------------------------
# SAVE RESULTS
# --------------------------------------
output_dir = "/Users/angusgray/Desktop/Dissertation/github_page/ALS_modelling_contralateral/saved_simulations"
os.makedirs(output_dir, exist_ok=True)

np.savez(
    output_dir + "/v0d_ramp_results.npz",
    weights=np.array(weights),
    peak_mnp1=np.array(rate_mnp1_peak),
    peak_mnp2=np.array(rate_mnp2_peak),
    spikes_L=np.array(block_spikes_L, dtype=object),
    spikes_R=np.array(block_spikes_R, dtype=object),
    v0d_L=np.array(block_spikes_V0d_L, dtype=object),
    v0d_R=np.array(block_spikes_V0d_R, dtype=object)
)
