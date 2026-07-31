
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
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib

ss.nest_start()
nn=netparams.neural_network()
conn=ConnectNetwork() 

import create_flx_rg as flx_rg
import create_ext_rg as ext_rg
import create_exc_inter_pop as exc
import create_inh_inter_pop as inh
import create_interneuron_pop as inter 
import create_mnp as mnp
import calculate_stability_metrics as calc
import send_receive_feedback as interface_fb

# ANALYSIS OF POPULATIONS  
# include Frequency of signal, Flx / Ext firing rate, amplitude, burst duration 
# Duty / On-off cycle, burst duration, cycle frequency, alternatation index - amount of overlap. 

def cpg_utils(nn,popfunc, conn,
              rg1, rg2, contra_rg1, contra_rg2,
              exc1, exc2, contra_exc1, contra_exc2, 
              V0c_1, V0c_2, 
              V1a_1, V1a_2,
              inh1, inh2, inh2_contra, 
              rc_1, rc_2, 
              mnp1, mnp2, contra_mnp1, contra_mnp2,
              V0d, V0d_contra, 
              V3_F, V3_F_contra,
              V3_E, V3_E_contra,
              label, 
              scale_log=None,           # optional
              ramp_weight_name=None,         # optional
              ramp_type=None):                   # optional 
    
    



    print("[DEBUG] Loaded cpg_data_utils from:", __file__)
    print(f"CPG UTILS {label} Started.")

    # Lazy folder creation — only create save dirs when we actually have output to write
    if nn.args.get("save_results", 0):
        import pathlib, yaml
        save_path = pathlib.Path(nn.path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "Figures").mkdir(exist_ok=True)
            with open(save_path / f"args_{nn.id_}.yaml", "w") as f:
                yaml.dump(nn.args, f)

    spike_count_array = []
    #Read spike data - rg populations
    senders_exc1,spiketimes_exc1 = popfunc.read_spike_data(rg1.spike_detector_rg_exc_bursting)
    senders_inh1,spiketimes_inh1 = popfunc.read_spike_data(rg1.spike_detector_rg_inh_bursting)
    senders_exc_tonic1,spiketimes_exc_tonic1 = popfunc.read_spike_data(rg1.spike_detector_rg_exc_tonic)
    senders_inh_tonic1,spiketimes_inh_tonic1 = popfunc.read_spike_data(rg1.spike_detector_rg_inh_tonic)

    senders_exc2,spiketimes_exc2 = popfunc.read_spike_data(rg2.spike_detector_rg_exc_bursting)
    senders_inh2,spiketimes_inh2 = popfunc.read_spike_data(rg2.spike_detector_rg_inh_bursting)
    senders_exc_tonic2,spiketimes_exc_tonic2 = popfunc.read_spike_data(rg2.spike_detector_rg_exc_tonic)
    senders_inh_tonic2,spiketimes_inh_tonic2 = popfunc.read_spike_data(rg2.spike_detector_rg_inh_tonic)

    # Read spike data - V2a excitatory interneurons
    senders_exc_inter_tonic1,spiketimes_exc_inter_tonic1 = popfunc.read_spike_data(exc1.spike_detector_exc_inter_tonic)
    senders_exc_inter_tonic2,spiketimes_exc_inter_tonic2 = popfunc.read_spike_data(exc2.spike_detector_exc_inter_tonic)

    # --- CONTRALATERAL V2a POPULATIONS ---
    senders_contra_exc_inter_tonic1, spiketimes_contra_exc_inter_tonic1 = popfunc.read_spike_data(contra_exc1.spike_detector_exc_inter_tonic)
    senders_contra_exc_inter_tonic2, spiketimes_contra_exc_inter_tonic2 = popfunc.read_spike_data(contra_exc2.spike_detector_exc_inter_tonic)

    #Read spike data - interneurons
    senders_V0c_1,spiketimes_V0c_1 = popfunc.read_spike_data(V0c_1.spike_detector)
    senders_V0c_2,spiketimes_V0c_2 = popfunc.read_spike_data(V0c_2.spike_detector)

    senders_V1a_1,spiketimes_V1a_1 = popfunc.read_spike_data(V1a_1.spike_detector)
    senders_V1a_2,spiketimes_V1a_2 = popfunc.read_spike_data(V1a_2.spike_detector)
    senders_rc_1,spiketimes_rc_1 = popfunc.read_spike_data(rc_1.spike_detector)
    senders_rc_2,spiketimes_rc_2 = popfunc.read_spike_data(rc_2.spike_detector)

    senders_V3_F_contra, spiketimes_V3_F_contra = popfunc.read_spike_data(V3_F_contra.spike_detector)
    senders_V3_F, spiketimes_V3_F = popfunc.read_spike_data(V3_F.spike_detector)

    senders_V3_E_contra, spiketimes_V3_E_contra = popfunc.read_spike_data(V3_E_contra.spike_detector)
    senders_V3_E, spiketimes_V3_E = popfunc.read_spike_data(V3_E.spike_detector)


    senders_V0d_contra, spiketimes_V0d_contra = popfunc.read_spike_data(V0d_contra.spike_detector)
    senders_V0d, spiketimes_V0d = popfunc.read_spike_data(V0d.spike_detector)
    
    # --- CONTRALATERAL RG POPULATIONS (full symmetry to ipsilateral) ---

    # Excitatory bursting
    senders_contra_rg_exc_burst1, spiketimes_contra_rg_exc_burst1 = popfunc.read_spike_data(contra_rg1.spike_detector_rg_exc_bursting)
    senders_contra_rg_exc_burst2, spiketimes_contra_rg_exc_burst2 = popfunc.read_spike_data(contra_rg2.spike_detector_rg_exc_bursting)

    # Inhibitory bursting
    senders_contra_rg_inh_burst1, spiketimes_contra_rg_inh_burst1 = popfunc.read_spike_data(contra_rg1.spike_detector_rg_inh_bursting)
    senders_contra_rg_inh_burst2, spiketimes_contra_rg_inh_burst2 = popfunc.read_spike_data(contra_rg2.spike_detector_rg_inh_bursting)

    # Excitatory tonic
    senders_contra_rg_exc_tonic1, spiketimes_contra_rg_exc_tonic1 = popfunc.read_spike_data(contra_rg1.spike_detector_rg_exc_tonic)
    senders_contra_rg_exc_tonic2, spiketimes_contra_rg_exc_tonic2 = popfunc.read_spike_data(contra_rg2.spike_detector_rg_exc_tonic)

    # Inhibitory tonic
    senders_contra_rg_inh_tonic1, spiketimes_contra_rg_inh_tonic1 = popfunc.read_spike_data(contra_rg1.spike_detector_rg_inh_tonic)
    senders_contra_rg_inh_tonic2, spiketimes_contra_rg_inh_tonic2 = popfunc.read_spike_data(contra_rg2.spike_detector_rg_inh_tonic)


    senders_contra_mnp1, spiketimes_contra_mnp1 = popfunc.read_spike_data(contra_mnp1.spike_detector_motor)
    senders_contra_mnp2, spiketimes_contra_mnp2 = popfunc.read_spike_data(contra_mnp2.spike_detector_motor)

    #Read spike data - MNPs
    senders_mnp1,spiketimes_mnp1 = popfunc.read_spike_data(mnp1.spike_detector_motor)
    senders_mnp2,spiketimes_mnp2 = popfunc.read_spike_data(mnp2.spike_detector_motor)


    # ========================== SPIKE REPORT CUSTOM POPFUNC FOR DEBUGGING TERMINAL INFO ==============================

     # spike sanity check 
    if nn.args['low_locomotion_v0d_right'] and nn.args['low_locomotion_v0d_left']:

        popfunc.spike_report("RG FLX IPSILATERAL", senders_rc_1,       spiketimes_rc_1)
        popfunc.spike_report("V0d IPSILATERAL",    senders_V0d,        spiketimes_V0d)
        
        popfunc.spike_report("RG FLX CONTRALATERAL EXC BURST", senders_contra_rg_exc_burst1, spiketimes_contra_rg_exc_burst1)
        popfunc.spike_report("V0d CONTRALATERAL",  senders_V0d_contra, spiketimes_V0d_contra)

        popfunc.spike_report("IPSILATERAL MNP FLX", senders_mnp1,spiketimes_mnp1)
        popfunc.spike_report("IPSILATERAL MNP EXT",  senders_mnp2, spiketimes_mnp2)

        popfunc.spike_report("CONTRALATERAL MNP FLX", senders_contra_mnp1, spiketimes_contra_mnp1)
        popfunc.spike_report("CONTRALATERAL MNP FLX", senders_contra_mnp2, spiketimes_contra_mnp2)
        

    if nn.fb_rg_flx == 1:
        #Read spike data - poisson generators
        senders_rg_flx_pg,spiketimes_rg_flx_pg = popfunc.read_spike_data(rg1.spike_detector_rg_flx_pg)

    if nn.rgs_connected==1:
    
            # inh1 = V2b 
            # inh2 = v1 
            # inh2_contra = v1_contra 
            #Read spike data - V1/V2b inhibitory populations
        senders_inh_inter_tonic1,spiketimes_inh_inter_tonic1 = popfunc.read_spike_data(inh1.spike_detector_inh_inter_tonic) 
        senders_inh_inter_tonic2,spiketimes_inh_inter_tonic2 = popfunc.read_spike_data(inh2.spike_detector_inh_inter_tonic)
        senders_inh_inter_tonic2_contra, spikestimes_inh_inter_tonic2_contra = popfunc.read_spike_data(inh2_contra.spike_detector_inh_inter_tonic)

    #Calculate synaptic balance of rg populations and total CPG network - missing interneurons
    if nn.calculate_balance==1:
            
        rg1_exc_burst_weight = conn.calculate_weighted_balance(rg1.rg_exc_bursting,rg1.spike_detector_rg_exc_bursting)
        rg1_inh_burst_weight = conn.calculate_weighted_balance(rg1.rg_inh_bursting,rg1.spike_detector_rg_inh_bursting)
        rg1_exc_tonic_weight = conn.calculate_weighted_balance(rg1.rg_exc_tonic,rg1.spike_detector_rg_exc_tonic)
        rg1_inh_tonic_weight = conn.calculate_weighted_balance(rg1.rg_inh_tonic,rg1.spike_detector_rg_inh_tonic)
        weights_per_pop1 = [rg1_exc_burst_weight,rg1_inh_burst_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight]
        absolute_weights_per_pop1 = [rg1_exc_burst_weight,abs(rg1_inh_burst_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight)]
        rg1_balance_pct = (sum(weights_per_pop1)/sum(absolute_weights_per_pop1))*100
        #print('RG1 balance %: ',round(rg1_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')
        
        rg2_exc_burst_weight = conn.calculate_weighted_balance(rg2.rg_exc_bursting,rg2.spike_detector_rg_exc_bursting)
        rg2_inh_burst_weight = conn.calculate_weighted_balance(rg2.rg_inh_bursting,rg2.spike_detector_rg_inh_bursting)
        rg2_exc_tonic_weight = conn.calculate_weighted_balance(rg2.rg_exc_tonic,rg2.spike_detector_rg_exc_tonic)
        rg2_inh_tonic_weight = conn.calculate_weighted_balance(rg2.rg_inh_tonic,rg2.spike_detector_rg_inh_tonic)
        weights_per_pop2 = [rg2_exc_burst_weight,rg2_inh_burst_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight]
        absolute_weights_per_pop2 = [rg2_exc_burst_weight,abs(rg2_inh_burst_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight)]
        rg2_balance_pct = (sum(weights_per_pop2)/sum(absolute_weights_per_pop2))*100
        #print('RG2 balance %: ',round(rg2_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')


        exc_tonic1_weight = conn.calculate_weighted_balance(exc1.exc_inter_tonic, exc1.spike_detector_exc_inter_tonic)
        exc_tonic2_weight = conn.calculate_weighted_balance(exc2.exc_inter_tonic, exc2.spike_detector_exc_inter_tonic)
        mnp1_weight = conn.calculate_weighted_balance(mnp1.motor_neuron_pop, mnp1.spike_detector_motor)
        mnp2_weight = conn.calculate_weighted_balance(mnp2.motor_neuron_pop, mnp2.spike_detector_motor)

        # --- SIDE 1 (Left hemisphere) ---
        weights_per_pop_side1 = [
            rg1_exc_burst_weight, rg1_inh_burst_weight,
            rg1_exc_tonic_weight, rg1_inh_tonic_weight,
            exc_tonic1_weight, mnp1_weight
        ]
        absolute_weights_per_pop_side1 = [
            abs(rg1_exc_burst_weight), abs(rg1_inh_burst_weight),
            abs(rg1_exc_tonic_weight), abs(rg1_inh_tonic_weight),
            abs(exc_tonic1_weight), abs(mnp1_weight)
        ]

        # --- SIDE 2 (Right hemisphere) ---
        weights_per_pop_side2 = [
            rg2_exc_burst_weight, rg2_inh_burst_weight,
            rg2_exc_tonic_weight, rg2_inh_tonic_weight,
            exc_tonic2_weight, mnp2_weight
        ]
        absolute_weights_per_pop_side2 = [
            abs(rg2_exc_burst_weight), abs(rg2_inh_burst_weight),
            abs(rg2_exc_tonic_weight), abs(rg2_inh_tonic_weight),
            abs(exc_tonic2_weight), abs(mnp2_weight)
        ]

        side1_balance_pct = (sum(weights_per_pop_side1) / sum(absolute_weights_per_pop_side1)) * 100
        side2_balance_pct = (sum(weights_per_pop_side2) / sum(absolute_weights_per_pop_side2)) * 100

        print('Balance % (RG1, RG2, Side1, Side2): ',
            round(rg1_balance_pct, 2),
            round(rg2_balance_pct, 2),
            round(side1_balance_pct, 2),
            round(side2_balance_pct, 2))

        if nn.rgs_connected == 1:
            inh1_weight = conn.calculate_weighted_balance(inh1.inh_pop, inh1.spike_detector_inh)
            inh2_weight = conn.calculate_weighted_balance(inh2.inh_pop, inh2.spike_detector_inh)

            weights_per_pop = [
                rg1_exc_burst_weight, rg1_inh_burst_weight, rg1_exc_tonic_weight, rg1_inh_tonic_weight,
                rg2_exc_burst_weight, rg2_inh_burst_weight, rg2_exc_tonic_weight, rg2_inh_tonic_weight,
                inh1_weight, inh2_weight,
                exc_tonic1_weight, mnp1_weight, exc_tonic2_weight, mnp2_weight
            ]

            absolute_weights_per_pop = [
                abs(rg1_exc_burst_weight), abs(rg1_inh_burst_weight), abs(rg1_exc_tonic_weight), abs(rg1_inh_tonic_weight),
                abs(rg2_exc_burst_weight), abs(rg2_inh_burst_weight), abs(rg2_exc_tonic_weight), abs(rg2_inh_tonic_weight),
                abs(inh1_weight), abs(inh2_weight),
                abs(exc_tonic1_weight), abs(mnp1_weight), abs(exc_tonic2_weight), abs(mnp2_weight)
            ]

            total_balance_pct = (sum(weights_per_pop) / sum(absolute_weights_per_pop)) * 100
            print('Balance % (complete network): ', round(total_balance_pct, 2),
                '>0 skew excitatory; <0 skew inhibitory')

    #Create Rate Coded Output
    if nn.rate_coded_plot==1:
        t_start = time.perf_counter()
        spike_bins_rg_exc1 = popfunc.rate_code_spikes(nn.flx_exc_bursting_count,spiketimes_exc1)
        spike_bins_rg_inh1 = popfunc.rate_code_spikes(nn.flx_inh_bursting_count,spiketimes_inh1)
        spike_bins_rg_exc_tonic1 = popfunc.rate_code_spikes(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
        spike_bins_rg_inh_tonic1 = popfunc.rate_code_spikes(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
        spike_bins_rg1 = spike_bins_rg_exc1+spike_bins_rg_exc_tonic1+spike_bins_rg_inh1+spike_bins_rg_inh_tonic1
        spike_bins_rg1_true = spike_bins_rg1
        print('Max spike count RG_F: ',max(spike_bins_rg1))
        spike_bins_rg1 = (spike_bins_rg1-np.min(spike_bins_rg1))/(np.max(spike_bins_rg1)-np.min(spike_bins_rg1))

        spike_bins_rg_exc2 = popfunc.rate_code_spikes(nn.ext_exc_bursting_count,spiketimes_exc2)
        spike_bins_rg_inh2 = popfunc.rate_code_spikes(nn.ext_inh_bursting_count,spiketimes_inh2)
        spike_bins_rg_exc_tonic2 = popfunc.rate_code_spikes(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
        spike_bins_rg_inh_tonic2 = popfunc.rate_code_spikes(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
        
        spike_bins_rg2 = spike_bins_rg_exc2+spike_bins_rg_exc_tonic2+spike_bins_rg_inh2+spike_bins_rg_inh_tonic2
        spike_bins_rg2_true = spike_bins_rg2
        print('Max spike count RG_E: ',max(spike_bins_rg2))
        spike_bins_rg2 = (spike_bins_rg2-np.min(spike_bins_rg2))/(np.max(spike_bins_rg2)-np.min(spike_bins_rg2))
        spike_bins_rgs = spike_bins_rg1+spike_bins_rg2

        spike_bins_exc_inter_tonic1 = popfunc.rate_code_spikes(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
        spike_bins_exc_inter1 = spike_bins_exc_inter_tonic1
        spike_bins_exc_inter1_true = spike_bins_exc_inter1
        
        spike_bins_exc_inter1 = (spike_bins_exc_inter1-np.min(spike_bins_exc_inter1))/(np.max(spike_bins_exc_inter1)-np.min(spike_bins_exc_inter1))
        spike_bins_exc_inter_tonic2 = popfunc.rate_code_spikes(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)
        spike_bins_exc_inter2 = spike_bins_exc_inter_tonic2
        spike_bins_exc_inter2_true = spike_bins_exc_inter2

        # V2a left
        print('Max spike count V2a Left: ', max(spike_bins_exc_inter1_true))
        print('Total spikes V2a Left: ', sum(spike_bins_exc_inter1_true))
        print('Mean spike count V2a Left: ', np.mean(spike_bins_exc_inter1_true))

        # V2a right
        print('Max spike count V2a Right: ', max(spike_bins_exc_inter2_true))
        print('Total spikes V2a Right: ', sum(spike_bins_exc_inter2_true))
        print('Mean spike count V2a Right: ', np.mean(spike_bins_exc_inter2_true))

        spike_bins_exc_inter2 = (spike_bins_exc_inter2-np.min(spike_bins_exc_inter2))/(np.max(spike_bins_exc_inter2)-np.min(spike_bins_exc_inter2))

        spike_bins_V0c_1 = popfunc.rate_code_spikes(nn.v0c_pop_size,spiketimes_V0c_1)
        spike_bins_V0c_1_true = spike_bins_V0c_1
        spike_bins_V0c_1 = (spike_bins_V0c_1-np.min(spike_bins_V0c_1))/(np.max(spike_bins_V0c_1)-np.min(spike_bins_V0c_1))
        spike_bins_V0c_2 = popfunc.rate_code_spikes(nn.v0c_pop_size,spiketimes_V0c_2)
        spike_bins_V0c_2_true = spike_bins_V0c_2  
        spike_bins_V0c_2 = (spike_bins_V0c_2-np.min(spike_bins_V0c_2))/(np.max(spike_bins_V0c_2)-np.min(spike_bins_V0c_2))
        spike_bins_V1a_1 = popfunc.rate_code_spikes(nn.v1a_pop_size,spiketimes_V1a_1)
        spike_bins_V1a_1_true = spike_bins_V1a_1
        spike_bins_V1a_1 = (spike_bins_V1a_1-np.min(spike_bins_V1a_1))/(np.max(spike_bins_V1a_1)-np.min(spike_bins_V1a_1))
        spike_bins_V1a_2 = popfunc.rate_code_spikes(nn.v1a_pop_size,spiketimes_V1a_2)
        spike_bins_V1a_2_true = spike_bins_V1a_2
        spike_bins_V1a_2 = (spike_bins_V1a_2-np.min(spike_bins_V1a_2))/(np.max(spike_bins_V1a_2)-np.min(spike_bins_V1a_2))
        spike_bins_rc_1 = popfunc.rate_code_spikes(nn.rc_pop_size,spiketimes_rc_1)
        spike_bins_rc_1_true = spike_bins_rc_1
        spike_bins_rc_1 = (spike_bins_rc_1-np.min(spike_bins_rc_1))/(np.max(spike_bins_rc_1)-np.min(spike_bins_rc_1))
        spike_bins_rc_2 = popfunc.rate_code_spikes(nn.rc_pop_size,spiketimes_rc_2)
        spike_bins_rc_2_true = spike_bins_rc_2
        spike_bins_rc_2 = (spike_bins_rc_2-np.min(spike_bins_rc_2))/(np.max(spike_bins_rc_2)-np.min(spike_bins_rc_2))
        
        spike_bins_mnp1 = popfunc.rate_code_spikes(nn.num_motor_neurons,spiketimes_mnp1)
        spike_bins_mnp1_true = spike_bins_mnp1

        print('Max spike count MNP FLX: ',max(spike_bins_mnp1_true))
        print('Total spike count MNP FLX: ', sum(spike_bins_mnp1_true))
        print('Mean spike count MNP FLX: ', np.mean(spike_bins_mnp1_true))


        spike_bins_mnp1 = (spike_bins_mnp1-np.min(spike_bins_mnp1))/(np.max(spike_bins_mnp1)-np.min(spike_bins_mnp1))
        spike_bins_mnp2 = popfunc.rate_code_spikes(nn.num_motor_neurons,spiketimes_mnp2)
        spike_bins_mnp2_true = spike_bins_mnp2

        print('Max spike count MNP EXT: ',max(spike_bins_mnp2_true))
        print('Total spike count MNP EXT: ', sum(spike_bins_mnp2_true))
        print('Mean spike count MNP EXT: ', np.mean(spike_bins_mnp2_true))

        spike_bins_mnp2 = (spike_bins_mnp2-np.min(spike_bins_mnp2))/(np.max(spike_bins_mnp2)-np.min(spike_bins_mnp2))
        spike_bins_mnps = spike_bins_mnp1+spike_bins_mnp2

        if nn.rgs_connected==1:
            spike_bins_inh_inter_tonic1 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1) # ISSUE how would we know whether this is _l or _r? this is a problem for plotting and calculations -- post meeting problem 
            spike_bins_inh_inter1 = spike_bins_inh_inter_tonic1
            
            spike_bins_inh_inter1_true = spike_bins_inh_inter1
            spike_bins_inh_inter1 = (spike_bins_inh_inter1-np.min(spike_bins_inh_inter1))/(np.max(spike_bins_inh_inter1)-np.min(spike_bins_inh_inter1))
            
            print("V2b total spikes:", np.sum(spike_bins_inh_inter1_true))

            spike_bins_inh_inter_tonic2 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v1_L,spiketimes_inh_inter_tonic2)
            spike_bins_inh_inter2 = spike_bins_inh_inter_tonic2
            
            spike_bins_inh_inter2_true = spike_bins_inh_inter2
            spike_bins_inh_inter2 = (spike_bins_inh_inter2-np.min(spike_bins_inh_inter2))/(np.max(spike_bins_inh_inter2)-np.min(spike_bins_inh_inter2)) # ISSUE how would we know whether this is _l or _r? this is a problem for plotting and calculations -- post meeting  problem 

            print("V1 total spikes:", np.sum(spike_bins_inh_inter2_true))
       
            
        t_stop = time.perf_counter()
        print('Rate coded activity complete, taking ',int(t_stop-t_start),' seconds.')

        print('[INFO] Starting Plotting Functions')

        #Plot rate-coded output
        t = np.arange(0,len(spike_bins_rg1),1)
    
        fig, ax = plt.subplots(4,sharex='all')
        ax[0].plot(t, spike_bins_V0c_1_true)
        ax[0].plot(t, spike_bins_V0c_2_true)
        ax[1].plot(t, spike_bins_V1a_1_true)
        ax[1].plot(t, spike_bins_V1a_2_true)		
        ax[2].plot(t, spike_bins_rc_1_true)
        ax[2].plot(t, spike_bins_rc_2_true) 
        ax[3].plot(t, spike_bins_mnp1_true)
        ax[3].plot(t, spike_bins_mnp2_true)
        for i in range(2):
            ax[i].set_xticks([])
            ax[i].set_xlim(0,len(spike_bins_rg1_true))
        ax[3].set_xlabel('Time (ms)')
        ax[3].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        ax[3].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        ax[3].set_xlim(0,len(spike_bins_rg1_true))
        ax[0].legend(['V0c_F', 'V0c_E'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['1a_F', '1a_E'],loc='upper right',fontsize='x-small') 
        ax[2].legend(['RC_F', 'RC_E'],loc='upper right',fontsize='x-small') 
        ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
        ax[0].set_title(f"Population output (V0c) - {label}")
        ax[1].set_title(f"Population output (1a) - {label}")
        ax[2].set_title(f"Population output (RC) - {label}")
        ax[3].set_title(f"Population output (MNP) - {label}")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
         # 
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_rate_coded_output_interneurons.png',bbox_inches="tight")
        
        fig, ax = plt.subplots(4,sharex='all')
        ax[0].plot(t, spike_bins_rg1_true)
        ax[0].plot(t, spike_bins_rg2_true)
        ax[1].plot(t, spike_bins_inh_inter1_true)
        ax[1].plot(t, spike_bins_inh_inter2_true)		
        ax[2].plot(t, spike_bins_exc_inter1_true)
        ax[2].plot(t, spike_bins_exc_inter2_true) 
        ax[3].plot(t, spike_bins_mnp1_true)
        ax[3].plot(t, spike_bins_mnp2_true)
        for i in range(2):
            ax[i].set_xticks([])
            ax[i].set_xlim(0,len(spike_bins_rg1_true))
        ax[3].set_xlabel('Time (ms)')
        ax[3].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        ax[3].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        ax[3].set_xlim(0,len(spike_bins_rg1_true))
        ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
        ax[2].legend(['V2a_F', 'V2a_E'],loc='upper right',fontsize='x-small') 
        ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
        ax[0].set_title(f"Population output (RG) - {label}")
        ax[1].set_title(f"Population output (V1/V2b) - {label}")
        ax[2].set_title(f"Population output (V2a) - {label}")
        ax[3].set_title(f"Population output (MNP) - {label}")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()


        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_rate_coded_output.png',bbox_inches="tight")

        fig, ax = plt.subplots(2,sharex='all')
        ax[0].plot(t, spike_bins_rg1_true)
        ax[0].set_ylabel('Frequency (Hz)')
        ax[0].plot(t, spike_bins_rg2_true)
        ax[1].plot(t, spike_bins_mnp1_true)
        ax[1].plot(t, spike_bins_mnp2_true)
        ax[0].set_xticks([])
        ax[0].set_xlim(0,len(spike_bins_rg1_true))
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Frequency (Hz)')
        ax[1].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        ax[1].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        ax[1].set_xlim(0,len(spike_bins_rg1_true))
        ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small')  
        ax[1].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
        ax[0].set_title(f"Population output (RG) - {label}")
        ax[1].set_title(f"P112 MNP Population Output - {label}")  # DEBUG  
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()

        if nn.args['save_as_svg'] and nn.args['save_results']:
            plt.savefig(nn.pathFigures + f'/{label}_rate_coded_output_rg_mnp.svg',
                        bbox_inches="tight", transparent=True)
        elif nn.args['save_results']:
            plt.savefig(nn.pathFigures + f'/{label}_rate_coded_output_rg_mnp.png',
                        dpi=300, bbox_inches="tight")

        plt.close(fig)
   
        if max(spike_bins_mnp1)>0 and max(spike_bins_mnp2)>0: 

            # calling analyze output function for MNP from calculate_stability_metrics.py 
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(spike_bins_mnp1,spike_bins_mnp2,spike_bins_mnp1_true,spike_bins_mnp2_true,'MNP',y_line_bd=0.4,y_line_phase=0.7)    

    if nn.isf_output==1:
        t_start = time.perf_counter()
        #Calculate instantaneous spiking frequency
        rgexc1_bursting_freq, rgexc1_bursting_times = popfunc.calculate_interspike_frequency(nn.flx_exc_bursting_count,spiketimes_exc1)
        rgexc2_bursting_freq, rgexc2_bursting_times = popfunc.calculate_interspike_frequency(nn.ext_exc_bursting_count,spiketimes_exc2)
        rgexc1_tonic_freq, rgexc1_tonic_times = popfunc.calculate_interspike_frequency(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
        rgexc2_tonic_freq, rgexc2_tonic_times = popfunc.calculate_interspike_frequency(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
        rginh1_bursting_freq, rginh1_bursting_times = popfunc.calculate_interspike_frequency(nn.flx_inh_bursting_count,spiketimes_inh1)
        rginh2_bursting_freq, rginh2_bursting_times = popfunc.calculate_interspike_frequency(nn.ext_inh_bursting_count,spiketimes_inh2)
        rginh1_tonic_freq, rginh1_tonic_times = popfunc.calculate_interspike_frequency(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
        rginh2_tonic_freq, rginh2_tonic_times = popfunc.calculate_interspike_frequency(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)


        # terminal information output  
        print("[INFO] FREQUENCY CALCULATIONS")
        print(f"RG1 Exc (bursting) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in rgexc1_bursting_freq]):.2f} Hz")
        print(f"RG2 Exc (bursting) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in rgexc2_bursting_freq]):.2f} Hz")
        print(f"RG1 Exc (tonic) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in rgexc1_tonic_freq]):.2f} Hz")
        print(f"RG2 Exc (tonic) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in rgexc2_tonic_freq]):.2f} Hz")

        print(f"RG1 Exc (bursting) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc1_bursting_freq]):.2f} Hz")
        print(f"RG2 Exc (bursting) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc2_bursting_freq]):.2f} Hz")
        print(f"RG1 Exc (tonic) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc1_tonic_freq]):.2f} Hz")
        print(f"RG2 Exc (tonic) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc2_tonic_freq]):.2f} Hz")

        print(f"RG1 Inh (bursting): {np.nanmean([np.nanmean(f) for f in rginh1_bursting_freq]):.2f} Hz")
        print(f"RG2 Inh (bursting): {np.nanmean([np.nanmean(f) for f in rginh2_bursting_freq]):.2f} Hz")

        v2a1_freq, v2a1_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
        v2a2_freq, v2a2_times =popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)

        print(f"V2a (Ipsilateral Flexor-Connected): {np.nanmean([np.nanmean(f) for f in v2a1_freq]):.2f} Hz")

        # --- CONTRALATERAL V2a frequencies ---
        contra_v2a1_freq, contra_v2a1_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic1)
        contra_v2a2_freq, contra_v2a2_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic2)

        print(f"V2a (Contralateral Flexor-Connected): {np.nanmean([np.nanmean(f) for f in contra_v2a1_freq]):.2f} Hz")
        
        v0c1_freq, v0c1_times = popfunc.calculate_interspike_frequency(nn.v0c_pop_size,spiketimes_V0c_1) # v0c1 - connecting to the flx
        v0c2_freq, v0c2_times = popfunc.calculate_interspike_frequency(nn.v0c_pop_size,spiketimes_V0c_2) # v0c2 - connecting to the ext. 

        # print(f"V0c1 flex freq: {v0c1_freq} and spike times: {v0c1_times}")
        # time.sleep(5)

        v0d_freq, v0d_times = popfunc.calculate_interspike_frequency(nn.v0d_pop_size,spiketimes_V0d)
        v3_F_freq, v3_times = popfunc.calculate_interspike_frequency(nn.v3_pop_size,spiketimes_V3_F)
        v3_E_freq, v3_times = popfunc.calculate_interspike_frequency(nn.v3_pop_size,spiketimes_V3_E)
        
        print(f"V0c (Flexor-Connected) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0c1_freq]):.2f} Hz")
        print(f"V0c (Extensor-Connected) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0c2_freq]):.2f} Hz")

        print(f"V0d (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0d_freq]):.2f} Hz")
        print(f"V3 (FLEXOR): {np.nanmean([np.nanmean(f) for f in v3_F_freq]):.2f} Hz")
        print(f"V3 (EXTENSOR): {np.nanmean([np.nanmean(f) for f in v3_F_freq]):.2f} Hz")

        v0d_contra_freq, v0d_contra_times = popfunc.calculate_interspike_frequency(nn.v0d_pop_size, spiketimes_V0d_contra)
        v3_F_contra_freq, v3_F_contra_times = popfunc.calculate_interspike_frequency(nn.v3_pop_size,spiketimes_V3_F_contra)
        v3_E_contra_freq, v3_E_contra_times = popfunc.calculate_interspike_frequency(nn.v3_pop_size,spiketimes_V3_E_contra)

        print(f"V0d (Contralateral): {np.nanmean([np.nanmean(f) for f in v0d_contra_freq]):.2f} Hz")
        print(f"V3 (Flexor): {np.nanmean([np.nanmean(f) for f in v3_F_contra_freq]):.2f} Hz")
        print(f"V0d (Extensor): {np.nanmean([np.nanmean(f) for f in v3_E_contra_freq]):.2f} Hz")
        
        v1a1_freq, v1a1_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_1)
        v1a2_freq, v1a2_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_2)
        
        rc1_freq, rc1_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_1)
        rc2_freq, rc2_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_2)
        
        if nn.rgs_connected:
            v2b_freq, v2b_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1) # issue here as well 

            v1_freq, v1_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1_L,spiketimes_inh_inter_tonic2)

            print(f"V1 (ipsilateral): {np.nanmean([np.nanmean(f) for f in v1_freq]):.2f} Hz")

            v1_contra_freq, v1_contra_times = popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1_L,spikestimes_inh_inter_tonic2_contra)
     
            print(f"V1 (contralateral): {np.nanmean([np.nanmean(f) for f in v1_contra_freq]):.2f} Hz")

            # Motor neuron instantaneous spiking frequencies
        mnp1_freq, mnp1_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_mnp1
        )
        mnp2_freq, mnp2_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_mnp2
        )

        print(f"Motor Neuron Pool (Ipsilateral) (Flexor): {np.nanmean([np.nanmean(f) for f in mnp1_freq]):.2f} Hz")
        print(f"Motor Neuron Pool (Ipsilateral) (Extensor): {np.nanmean([np.nanmean(f) for f in mnp2_freq]):.2f} Hz")

        mnp1_contra_freq, mnp1_contra_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_contra_mnp1)
        
        mnp2_contra_freq, mnp2_contra_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_contra_mnp2)
        
        print(f"Motor Neuron Pool (Contralateral) (Flexor): {np.nanmean([np.nanmean(f) for f in mnp1_contra_freq]):.2f} Hz")
        print(f"Motor Neuron Pool (Contralateral) (Extensor): {np.nanmean([np.nanmean(f) for f in mnp2_contra_freq]):.2f} Hz")
        
        t_stop = time.perf_counter()    
        
        # calculating ISF - Inter Spike Frequency. 
        print('Calculating Inter-Spiking Frequency complete, taking ',int(t_stop-t_start),' seconds.')

        print('Moving onto Convolving Spiking activity Processing (Taking a spike train (series of discrete spikes) and smoothing it into a continuous signal)')
        
        t_start = time.perf_counter()
        #Convolve spike data - RG populations

        # print("[DEBUG] flx_exc_tonic_count =", nn.flx_exc_tonic_count)
        # print("[DEBUG] spiketimes_exc_tonic1 type:", type(spiketimes_exc_tonic1))
        # print("[DEBUG] spiketimes_exc_tonic1 example:", spiketimes_exc_tonic1[:10])
        
        rg_exc_tonic_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
        rg_inh_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_inh_bursting_count,spiketimes_inh1)
        rg_inh_tonic_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
        rg_exc_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count, spiketimes_exc1)
    
        rg1_convolved = np.vstack([rg_exc_convolved1,rg_inh_convolved1])
        rg1_convolved = np.vstack([rg1_convolved,rg_exc_tonic_convolved1])
        rg1_convolved = np.vstack([rg1_convolved,rg_inh_tonic_convolved1])
        rg1_convolved = rg1_convolved.mean(axis=0)

        rg_exc_convolved2, _, _  = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count,spiketimes_exc2)
        rg_exc_tonic_convolved2, _, _  = popfunc.convolve_spiking_activity(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
        rg_inh_convolved2, _, _  = popfunc.convolve_spiking_activity(nn.ext_inh_bursting_count,spiketimes_inh2)
        rg_inh_tonic_convolved2, _, _  = popfunc.convolve_spiking_activity(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
        rg2_convolved = np.vstack([rg_exc_convolved2,rg_inh_convolved2])
        rg2_convolved = np.vstack([rg2_convolved,rg_exc_tonic_convolved2])
        rg2_convolved = np.vstack([rg2_convolved,rg_inh_tonic_convolved2])
        rg2_convolved = rg2_convolved.mean(axis=0)

        #Convolve spike data - V2a excitatory interneuron populations
        v2a1_convolved, _, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
        v2a2_convolved, _, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)

        # --- CONTRALATERAL V2a convolved ---
        v2a_contra1_convolved, _, _ = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic1)
        v2a_contra2_convolved, _, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic2)

        # Average contra V2a if needed (match ipsi structure)
        v2a_contra_convolved = np.vstack([v2a_contra1_convolved, v2a_contra2_convolved]).mean(axis=0)

        # can be used when importing in contralateral populations 
        #if nn.contralateral_projections_v0c_right and nn.contralateral_projections_v0c_left:

        # both ipsilateral flexor and extensor populations here. 
        v0c1_convolved, _, _  = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_1)
        v0c2_convolved, _, _  = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_2)

         
        v1a1_convolved, _, _ = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_1)
        v1a2_convolved, _, _ = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_2)
        
        rc1_convolved, _, _ = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_1)
        rc2_convolved, _, _ = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_2)

        #Convolve spike data - v2a 

        if nn.low_locomotion_v3_left and nn.low_locomotion_v3_right:

            v3_F_convolved, _, v3_F_neuron_convolved = popfunc.convolve_spiking_activity(nn.v3_pop_size,spiketimes_V3_F)   # population-averaged signal (T, )
            v3_F_contra_convolved, _, v3_F_contra_neuron_convolved = popfunc.convolve_spiking_activity(nn.v3_pop_size, spiketimes_V3_F_contra)

            v3_E_convolved, _, v3_E_neuron_convolved = popfunc.convolve_spiking_activity(nn.v3_pop_size,spiketimes_V3_E)   # population-averaged signal (T, )
            v3_E_contra_convolved, _, v3_E_contra_neuron_convolved = popfunc.convolve_spiking_activity(nn.v3_pop_size, spiketimes_V3_E_contra)

        if nn.low_locomotion_v0d_left and nn.low_locomotion_v0d_right: 
        
            v0d_contra_convolved, _, v0d_contra_neuron_convolved = popfunc.convolve_spiking_activity(nn.v0d_pop_size, spiketimes_V0d_contra)
            v0d_convolved, _, v0d_neuron_convolved = popfunc.convolve_spiking_activity(nn.v0d_pop_size,spiketimes_V0d)

        #Convolve spike data - MNPs
        mnp1_convolved, convolved_time, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp1)
        mnp2_convolved, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp2)

        mnp1_convolved_contra, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_contra_mnp1)
        mnp2_convolved_contra, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_contra_mnp2)

        # --- Convolve CONTRALATERAL RG populations (full symmetry to ipsilateral) ---



        rgexc1_bursting_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.flx_exc_bursting_count, spiketimes_contra_rg_exc_burst1)
        rgexc2_bursting_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.ext_exc_bursting_count, spiketimes_contra_rg_exc_burst2)

        rgexc1_tonic_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.flx_exc_tonic_count, spiketimes_contra_rg_exc_tonic1)
        rgexc2_tonic_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.ext_exc_tonic_count, spiketimes_contra_rg_exc_tonic2)

        rginh1_bursting_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.flx_inh_bursting_count, spiketimes_contra_rg_inh_burst1)
        rginh2_bursting_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.ext_inh_bursting_count, spiketimes_contra_rg_inh_burst2)

        rginh1_tonic_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
        rginh2_tonic_freq_contra, _ = popfunc.calculate_interspike_frequency(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)


        # Excitatory bursting
        contra_rg_exc_burst_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count, spiketimes_contra_rg_exc_burst1)
        contra_rg_exc_burst_convolved2, _, _ = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count, spiketimes_contra_rg_exc_burst2)

        # Inhibitory bursting
        contra_rg_inh_burst_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_inh_bursting_count, spiketimes_contra_rg_inh_burst1)
        contra_rg_inh_burst_convolved2, _, _ = popfunc.convolve_spiking_activity(nn.ext_inh_bursting_count, spiketimes_contra_rg_inh_burst2)

        # Excitatory tonic
        contra_rg_exc_tonic_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_exc_tonic_count, spiketimes_contra_rg_exc_tonic1)
        contra_rg_exc_tonic_convolved2, _, _ = popfunc.convolve_spiking_activity(nn.ext_exc_tonic_count, spiketimes_contra_rg_exc_tonic2)

        # Inhibitory tonic
        contra_rg_inh_tonic_convolved1, _, _ = popfunc.convolve_spiking_activity(nn.flx_inh_tonic_count, spiketimes_contra_rg_inh_tonic1)
        contra_rg_inh_tonic_convolved2, _, _ = popfunc.convolve_spiking_activity(nn.ext_inh_tonic_count, spiketimes_contra_rg_inh_tonic2)

        
        print("[INFO] FREQUENCY CALCULATIONS")
        contra_rg_flx_freq = np.nanmean([np.nanmean(f) for f in rgexc1_bursting_freq_contra])

        print(f"RG1 Exc (bursting) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc1_bursting_freq_contra]):.2f} Hz")
        print(f"RG2 Exc (bursting) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc2_bursting_freq_contra]):.2f} Hz")
        print(f"RG1 Exc (tonic) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc1_tonic_freq_contra]):.2f} Hz")
        print(f"RG2 Exc (tonic) (Contralateral): {np.nanmean([np.nanmean(f) for f in rgexc2_tonic_freq_contra]):.2f} Hz")

        # Average all subpops (same logic as ipsilateral)
        contra_rg1_convolved = np.vstack([
            contra_rg_exc_burst_convolved1,
            contra_rg_inh_burst_convolved1,
            contra_rg_exc_tonic_convolved1,
            contra_rg_inh_tonic_convolved1
        ]).mean(axis=0)

        contra_rg2_convolved = np.vstack([
            contra_rg_exc_burst_convolved2,
            contra_rg_inh_burst_convolved2,
            contra_rg_exc_tonic_convolved2,
            contra_rg_inh_tonic_convolved2
        ]).mean(axis=0)


        # Convolve contralateral MNP
        contra_mnp1_convolved, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, spiketimes_contra_mnp1)
        contra_mnp2_convolved, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, spiketimes_contra_mnp2)

                
        # Convolve spike data - inh populations
        if nn.rgs_connected == 1:
            v2b_convolved, _, _  = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v2b, spiketimes_inh_inter_tonic1)
            v1_convolved, _, _  = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v1, spiketimes_inh_inter_tonic2)
            v1_contra_convolved, _, _ = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v1, spikestimes_inh_inter_tonic2_contra)


        t_stop = time.perf_counter()    
        print('Convolved spiking activity complete, taking ',int(t_stop-t_start),' seconds.')

        rg1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rgexc1_bursting_freq]))
        rg2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rgexc2_bursting_freq]))
        rg1_conv_max = np.nanmax(rg1_convolved)
        rg2_conv_max = np.nanmax(rg2_convolved)
        rg1_scale = rg1_isf_max / rg1_conv_max
        rg2_scale = rg2_isf_max / rg2_conv_max
        
        print('Max firing rate of a Flx RG (ISF):',round(rg1_isf_max,2),'Ext RG:',round(rg2_isf_max,2))
        print('Max firing rate of a Flx RG (Convolved):',round(rg1_conv_max,2),'Ext RG:',round(rg2_conv_max,2))
        
        print('Convolved max is',round(rg1_scale,3),round(rg2_scale,3), 'times the size of ISF max (Flx, Ext).')

        
        v2b_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2b_freq]))
        v1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1_freq]))
        v1_contra_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1_contra_freq]))
        
        v2b_conv_max = np.nanmax(v2b_convolved)
        v1_conv_max = np.nanmax(v1_convolved)
        v1_contra_conv_max = np.nanmax(v1_contra_convolved)
                
        v2b_scale = v2b_isf_max / v2b_conv_max
     
        v1_scale = v1_isf_max / v1_conv_max
        v1_scale_contra = v1_contra_isf_max / v1_contra_conv_max 

        print('Max Firing rate of V1 (Ipsilateral) (ISF):', round(v1_isf_max,2), ' Max Firing rate of V1 (Contralateral) (ISF)', round(v1_contra_isf_max,2))
        
        v2a1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2a1_freq]))
        v2a2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v2a2_freq]))
        v2a1_conv_max = np.nanmax(v2a1_convolved)
        v2a2_conv_max = np.nanmax(v2a2_convolved)
        v2a1_scale = v2a1_isf_max / v2a1_conv_max
        v2a2_scale = v2a2_isf_max / v2a2_conv_max

        # --- CONTRALATERAL V2a scaling ---
        contra_v2a1_isf_max = np.nanmax([np.nanmean(f) for f in contra_v2a1_freq])
        contra_v2a2_isf_max = np.nanmax([np.nanmean(f) for f in contra_v2a2_freq])

        contra_v2a1_conv_max = np.nanmax(v2a_contra1_convolved)
        contra_v2a2_conv_max = np.nanmax(v2a_contra2_convolved)

        contra_v2a1_scale = contra_v2a1_isf_max / contra_v2a1_conv_max
        contra_v2a2_scale = contra_v2a2_isf_max / contra_v2a2_conv_max

        # For averaged contra V2a signal
        contra_v2a_scale = np.nanmax([contra_v2a1_scale, contra_v2a2_scale])

        # can be used for contralateral. 
        #if nn.contralateral_projections_v0c_left and nn.contralateral_projections_v0c_right: 

        # ipsilateral population here. 
        v0c1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0c1_freq]))
        v0c2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0c2_freq]))
        v0c1_conv_max = np.nanmax(v0c1_convolved)
        v0c2_conv_max = np.nanmax(v0c2_convolved)
        v0c1_scale = v0c1_isf_max / v0c1_conv_max
        v0c2_scale = v0c2_isf_max / v0c2_conv_max

        if nn.low_locomotion_v3_left and nn.low_locomotion_v3_right:

            # FLEXOR - FLEXOR 
            v3_F_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v3_F_freq]))
            v3_F_conv_max = np.nanmax(v3_F_convolved)
            v3_F_contra_isf_max = np.nanmax([np.nanmean(f) for f in v3_F_contra_freq])

            print('Max Firing rate of V3 FLEXOR (Ipsilateral) (ISF):', round(v3_F_isf_max,2), ' Max Firing rate of V3 FLEXOR (Contralateral) (ISF)', round(v3_F_contra_isf_max,2))

            v3_F_contra_scale = v3_F_contra_isf_max / np.nanmax(v3_F_contra_convolved)
            v3_F_scale = v3_F_isf_max / v3_F_conv_max

            v3_F_convolved_scaled = v3_F_scale * v3_F_convolved
            v3_F_contra_convolved_scaled = v3_F_contra_scale * v3_F_contra_convolved

            # EXTENSOR - EXTENSOR 
            v3_E_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v3_E_freq]))
            v3_E_conv_max = np.nanmax(v3_E_convolved)
            v3_E_contra_isf_max = np.nanmax([np.nanmean(f) for f in v3_E_contra_freq])

            print('Max Firing rate of V3 EXTENSOR (Ipsilateral) (ISF):', round(v3_E_isf_max,2), ' Max Firing rate of V3 EXTENSOR (Contralateral) (ISF)', round(v3_E_contra_isf_max,2))

            v3_E_contra_scale = v3_E_contra_isf_max / np.nanmax(v3_E_contra_convolved)
            v3_E_scale = v3_E_isf_max / v3_E_conv_max

            v3_E_convolved_scaled = v3_E_scale * v3_E_convolved
            v3_E_contra_convolved_scaled = v3_E_contra_scale * v3_E_contra_convolved
       
        # V0d firing metric output
        if nn.low_locomotion_v0d_left and nn.low_locomotion_v0d_right: 
       
            v0d_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0d_freq]))
            v0d_contra_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0d_contra_freq]))
            V0d_conv_max = np.nanmax(v0d_convolved)
            V0d_contra_conv_max = np.nanmax(v0d_contra_convolved)

            print('Max Firing rate of V0D (Ipsilateral) (ISF):', round(v0d_isf_max,2), ' Max Firing rate of V0D (Contralateral) (ISF)', round(v0d_contra_isf_max,2))
            
            v0d_scale = v0d_isf_max / V0d_conv_max
            v0d_contra_scale = v0d_contra_isf_max / V0d_contra_conv_max
            
            v0d_convolved_scaled = v0d_scale * v0d_convolved
            v0d_contra_convolved_scaled = v0d_contra_scale * v0d_contra_convolved

            v0d_ipsi_y   = v0d_convolved * v0d_scale
            v0d_contra_y = v0d_contra_convolved * v0d_contra_scale

            # choose a window + with burn. 
            t0 = float(t[0] + 50.0)
            t1 = float(t[-1])

            # defining peak params to gather some metrics on V0d activity. 
            MIN_PEAK_HEIGHT   = 30.0
            MIN_PEAK_DIST_MS  = 147

            pm_ipsi = popfunc.window_peak_rate_metrics(
                t, v0d_ipsi_y, t0, t1,
                min_peak_height=MIN_PEAK_HEIGHT,
                min_peak_distance_ms=MIN_PEAK_DIST_MS
            )

            pm_contra = popfunc.window_peak_rate_metrics(
                t, v0d_contra_y, t0, t1,
                min_peak_height=MIN_PEAK_HEIGHT,
                min_peak_distance_ms=MIN_PEAK_DIST_MS
            )

            # These key names depend on your popfunc implementation.
            # Common patterns: "peak_heights", or "peak_values".
            ipsi_heights = np.asarray(
                pm_ipsi.get("peak_heights", pm_ipsi.get("peak_values", [])),
                dtype=float
            )
            contra_heights = np.asarray(
                pm_contra.get("peak_heights", pm_contra.get("peak_values", [])),
                dtype=float
            )

            v0d_ipsi_peak_mean   = float(np.nanmean(ipsi_heights)) if ipsi_heights.size else np.nan
            v0d_contra_peak_mean = float(np.nanmean(contra_heights)) if contra_heights.size else np.nan

            # NaN output? 
            print(f'Average Peak of V0D (Ipsilateral) : {v0d_ipsi_peak_mean} Average Peak of V0D (Contralateral): {v0d_contra_peak_mean} ')


        v1a1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1a1_freq]))
        v1a2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v1a2_freq]))
        v1a1_conv_max = np.nanmax(v1a1_convolved)
        v1a2_conv_max = np.nanmax(v1a2_convolved)
        v1a1_scale = v1a1_isf_max / v1a1_conv_max
        v1a2_scale = v1a2_isf_max / v1a2_conv_max
        
        rc1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rc1_freq]))
        rc2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in rc2_freq]))
        rc1_conv_max = np.nanmax(rc1_convolved)
        rc2_conv_max = np.nanmax(rc2_convolved)
        rc1_scale = rc1_isf_max / rc1_conv_max
        rc2_scale = rc2_isf_max / rc2_conv_max
        
        mnp1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in mnp1_freq]))
        mnp2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in mnp2_freq]))
        mnp1_conv_max = np.nanmax(mnp1_convolved)
        mnp2_conv_max = np.nanmax(mnp2_convolved)
        mnp1_scale = mnp1_isf_max / mnp1_conv_max
        mnp2_scale = mnp2_isf_max / mnp2_conv_max

        print('Max firing rate of a MNP FLX (ISF):',round(mnp1_isf_max,2),'MNP EXT:',round(mnp2_isf_max,2))
        print('Max firing rate of a MNP FLX (Convolved):',round(mnp1_conv_max,2),'MNP EXT:',round(mnp2_conv_max,2))
       
        # Contralateral RG ISF (use excitatory bursting subpopulation, symmetric to ipsi)
        contra_rg1_freq, contra_rg1_times = popfunc.calculate_interspike_frequency(
            nn.flx_exc_bursting_count, spiketimes_contra_rg_exc_burst1
        )

        contra_rg2_freq, contra_rg2_times = popfunc.calculate_interspike_frequency(
            nn.ext_exc_bursting_count, spiketimes_contra_rg_exc_burst2
        )

        contra_mnp1_freq, contra_mnp1_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_contra_mnp1)

        contra_mnp2_freq, contra_mnp2_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_contra_mnp2)

        contra_rg1_conv_max = np.nanmax(contra_rg1_convolved)
        contra_rg2_conv_max = np.nanmax(contra_rg2_convolved)

        contra_rg1_isf_max = np.nanmax([np.nanmean(f) for f in contra_rg1_freq])
        contra_rg2_isf_max = np.nanmax([np.nanmean(f) for f in contra_rg2_freq])
        contra_mnp1_isf_max = np.nanmax([np.nanmean(f) for f in contra_mnp1_freq])
        contra_mnp2_isf_max = np.nanmax([np.nanmean(f) for f in contra_mnp2_freq])

        contra_rg1_scale = contra_rg1_isf_max / np.nanmax(contra_rg1_convolved)
        contra_rg2_scale = contra_rg2_isf_max / np.nanmax(contra_rg2_convolved)
        contra_mnp1_scale = contra_mnp1_isf_max / np.nanmax(contra_mnp1_convolved)
        contra_mnp2_scale = contra_mnp2_isf_max / np.nanmax(contra_mnp2_convolved)

        contra_rg1_scale = contra_rg1_isf_max / contra_rg1_conv_max
        contra_rg2_scale = contra_rg2_isf_max / contra_rg2_conv_max
        
        print('Max firing rate of a Flx MN (ISF):',round(mnp1_isf_max,2),'Ext MN:',round(mnp2_isf_max,2))
        print('Max firing rate of a Flx MN (Convolved):',round(mnp1_conv_max,2),'Ext MN:',round(mnp2_conv_max,2))
        print('Convolved max is',round(mnp1_scale,3),round(mnp2_scale,3), 'times the size of ISF max (Flx, Ext).')
        
        # Scaling Rg1 & Rg2 
        mnp1_convolved_scaled = mnp1_convolved * mnp1_scale
        mnp2_convolved_scaled = mnp2_convolved * mnp2_scale
        mnp1_convolved_scaled_mean = np.nanmean(mnp1_convolved_scaled)
        mnp2_convolved_scaled_mean = np.nanmean(mnp2_convolved_scaled)
        mnp1_convolved_max_scaled = np.nanmax(mnp1_convolved * mnp1_scale)
        mnp2_convolved_max_scaled = np.nanmax(mnp2_convolved * mnp2_scale)
        

        if nn.healthy_regime_metrics == 1: 
            # --- Canonical timebase for convolved signals ---
            t_ms = np.asarray(convolved_time, dtype=float)

            # --- Scaled RG traces (define ONCE) ---
            rgF_ipsi  = rg1_convolved * rg1_scale
            rgE_ipsi  = rg2_convolved * rg2_scale
            rgF_contra = contra_rg1_convolved * contra_rg1_scale
            rgE_contra = contra_rg2_convolved * contra_rg2_scale

            # optional sanity prints
            print("[INFO] RG ipsi max:", np.nanmax(rgF_ipsi), np.nanmax(rgE_ipsi))
            print("[INFO] RG contra max:", np.nanmax(rgF_contra), np.nanmax(rgE_contra))

            if nn.healthy_regime_metrics == 1:
                healthy_run = {
                    "t_ms": t_ms,
                    "rgF_ipsi": rgF_ipsi,
                    "rgE_ipsi": rgE_ipsi,
                    "rgF_contra": rgF_contra,
                    "rgE_contra": rgE_contra,
                }

                config = popfunc.calibrate_rg_regime_from_healthy([healthy_run])
                print("\n[AUTO-DERIVED RG REGIME CONFIG (single run starter)]")
                for k, v in config.items():
                    print(f"  {k}: {v}")

                windows = popfunc.rg_regime_metrics_over_windows(
                    t_ms,
                    healthy_run["rgF_ipsi"], healthy_run["rgE_ipsi"],
                    check_window_ms=config["CHECK_WINDOW_MS"],
                    min_peak_distance_ms=config["MIN_PEAK_DIST_MS"],
                    thresh_frac=0.2,
                    adaptive_peaks=True,
                )

                print("\n[RG IPSI WINDOW METRICS]")
                for w in windows[:10]:
                    (w0, w1) = w["window"]
                    F = w["RG_F"]; E = w["RG_E"]
                    print(
                        f"  {w0:.0f}-{w1:.0f} ms | "
                        f"F: n_peaks={F['n_peaks_peaks']} med_ipi={F['median_ipi_ms']:.1f} cv={F['cv_ipi']:.2f} "
                        f"| E: n_peaks={E['n_peaks_peaks']} med_ipi={E['median_ipi_ms']:.1f} cv={E['cv_ipi']:.2f}"
                    )


                    # this is what we are wanting to find out...  
                    # bin_ms           = float(nn.BIN_MS)
                    # min_peaks        = int(nn.MIN_PEAKS)
                    # min_peak_dist_ms = float(nn.MIN_PEAK_DIST_MS)
                    # prominence       = float(nn.PEAK_STRENGTH)    
                    # max_median_ipi   = float(nn.MAX_MEDIAN_IPI_MS)
                    # max_cv           = float(nn.MAX_CV)
                    # check_window_ms  = float(nn.CHECK_WINDOW_MS)

        print('[INFO] After scaling max firing rate of a Flx MN (Convolved):',round(mnp1_convolved_max_scaled,2),'Ext MN:',round(mnp2_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx MN (Convolved):',round(mnp1_convolved_scaled_mean,2),'Ext MN:',round(mnp2_convolved_scaled_mean,2))
        
          # prepping variables for analyze_output function 
        mnp1_avg_norm = (mnp1_convolved-np.min(mnp1_convolved))/(np.max(mnp1_convolved)-np.min(mnp1_convolved))
        mnp2_avg_norm = (mnp2_convolved-np.min(mnp2_convolved))/(np.max(mnp2_convolved)-np.min(mnp2_convolved))

        
        # Scaling mnp1 & mnp2 contra 
        mnp1_contra_convolved_scaled = mnp1_convolved_contra * mnp1_scale
        mnp2_contra_convolved_scaled = mnp2_convolved_contra * mnp2_scale
        
        mnp1_contra_convolved_scaled_mean = np.nanmean(mnp1_contra_convolved_scaled)
        mnp2_contra_convolved_scaled_mean = np.nanmean(mnp2_contra_convolved_scaled)
        mnp1_contra_convolved_max_scaled = np.nanmax(mnp1_convolved_contra * mnp1_scale)
        mnp2_conta_convolved_max_scaled = np.nanmax(mnp2_convolved_contra * mnp1_scale)

        print('[INFO] After scaling max firing rate of a Flx CONTRA  MN (Convolved):',round(mnp1_contra_convolved_max_scaled,2),'Ext MN:',round(mnp2_conta_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx CONTRA MN (Convolved):',round(mnp1_contra_convolved_scaled_mean,2),'Ext MN:',round(mnp2_contra_convolved_scaled_mean,2))
        
        mnp1_contra_avg_norm = (mnp1_convolved_contra-np.min(mnp1_convolved_contra))/(np.max(mnp1_convolved_contra)-np.min(mnp1_convolved_contra))
        mnp2_contra_avg_norm = (mnp2_convolved_contra-np.min(mnp2_convolved_contra))/(np.max(mnp2_convolved_contra)-np.min(mnp2_convolved_contra))

         # Scaling Rg1 & Rg2
        rg1_convolved_scaled = rg1_convolved * rg1_scale
        rg2_convolved_scaled = rg2_convolved * rg2_scale
        rg1_convolved_scaled_mean = np.nanmean(rg1_convolved_scaled)
        rg_2_convolved_scaled_mean = np.nanmean(rg2_convolved_scaled)
        rg_1_convolved_max_scaled = np.nanmax(rg1_convolved * rg1_scale)
        rg_2_convolved_max_scaled = np.nanmax(rg2_convolved * rg2_scale)

        print('[INFO] After scaling max firing rate of a Flx RG (Convolved):',round(rg_1_convolved_max_scaled,2),'Ext RG:',round(rg_2_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx MN (Convolved):',round(rg1_convolved_scaled_mean,2),'Ext MN:',round(rg_2_convolved_scaled_mean,2))
        
        rg1_avg_norm = (rg1_convolved-np.min(rg1_convolved))/(np.max(rg1_convolved)-np.min(rg1_convolved))
        rg2_avg_norm = (rg2_convolved-np.min(rg2_convolved))/(np.max(rg2_convolved)-np.min(rg2_convolved))

        rg1_contra_convolved_scaled = contra_rg1_convolved * rg1_scale 
        rg2_contra_convolved_scaled = contra_rg2_convolved * rg2_scale 
        rg1_contra_convolved_scaled_mean = np.nanmean(rg1_contra_convolved_scaled)
        rg2_contra_convolved_scaled_mean = np.nanmean(rg2_contra_convolved_scaled)
        rg_1_contra_convolved_max_scaled = np.nanmax(contra_rg1_convolved * rg1_scale)
        rg_2_contra_convolved_max_scaled = np.nanmax(contra_rg2_convolved * rg2_scale) 
        
        # The highest instantaneous population firing rate reached at any moment during the analysed time window, after spike binning + smoothing.
        print('[INFO] After scaling max firing rate of a Flx RG (Convolved):',round(rg_1_contra_convolved_max_scaled,2),'Ext RG:',round(rg_2_contra_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx MN (Convolved):',round(rg1_contra_convolved_scaled_mean,2),'Ext MN:',round(rg2_contra_convolved_scaled_mean,2))
        
        rg1_contra_avg_norm = (contra_rg1_convolved-np.min(contra_rg1_convolved))/(np.max(contra_rg1_convolved)-np.min(contra_rg1_convolved))
        rg2_contra_avg_norm = (contra_rg2_convolved-np.min(contra_rg2_convolved))/(np.max(contra_rg2_convolved)-np.min(contra_rg2_convolved))

        # Scaling rg1 and rg2 contra 
        if max(mnp1_avg_norm)>0 and max(mnp2_avg_norm)>0: 

            print("[INFO] Calling Analyze Output Function for MNP (Ipsilateral) ")
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(mnp1_avg_norm,mnp2_avg_norm,mnp1_convolved_scaled,mnp2_convolved_scaled,'MNP',y_line_bd=0.4,y_line_phase=0.7)
            print(f"[INFO] Avg Frequency: {avg_freq} , Avg Phase: {avg_phase}, Burst Duration Comparision: {bd_comparison} ")

        if max(mnp1_contra_avg_norm) > 0 and max(mnp2_contra_avg_norm) > 0: 

            print("[INFO] Calling Analyze Output Function for MNP (Contralateral)")
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(mnp1_avg_norm,mnp2_avg_norm,mnp1_convolved_scaled,mnp2_convolved_scaled,'MNP',y_line_bd=0.4,y_line_phase=0.7)
            print(f"[INFO] Avg Frequency: {avg_freq} , Avg Phase: {avg_phase}, Burst Duration Comparision: {bd_comparison} ")
    
       
        if max(rg1_avg_norm) > 0 and max(rg2_avg_norm) > 0:  

            print("[INFO] Calling Analyze Output Function for RG (Ipsilateral)")
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(rg1_avg_norm,rg2_avg_norm,rg1_convolved_scaled,rg2_convolved_scaled,'RG',y_line_bd=0.4,y_line_phase=0.7)
            print(f"[INFO] Avg Frequency: {avg_freq} , Avg Phase: {avg_phase}, Burst Duration Comparision: {bd_comparison} ")

        if max(rg1_contra_avg_norm) > 0 and max(rg2_contra_avg_norm) > 0: 

            print("[INFO] Calling Analyze Output Function for RG (Contralateral)")
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(rg1_contra_avg_norm,rg2_contra_avg_norm,rg1_contra_convolved_scaled,rg2_contra_convolved_scaled,'RG',y_line_bd=0.4,y_line_phase=0.7)
            print(f"[INFO] Avg Frequency: {avg_freq} , Avg Phase: {avg_phase}, Burst Duration Comparision: {bd_comparison} ")
            
        
        t = convolved_time
        xticks = np.arange(start=np.ceil(t[0] / 1000) * 1000, stop=t[-1], step=1000)
        fig, ax = plt.subplots(4,sharex='all',figsize=(18, 12))    
        ax[0].plot(t, rg1_convolved* rg1_scale)
        ax[0].plot(t, rg2_convolved* rg2_scale)
        ax[1].plot(t, v2b_convolved* v2b_scale)
        ax[1].plot(t, v1_convolved* v1_scale)
        ax[2].plot(t, v2a1_convolved* v2a1_scale)
        ax[2].plot(t, v2a2_convolved* v2a2_scale)
        ax[3].plot(t, mnp1_convolved* mnp1_scale)
        ax[3].plot(t, mnp2_convolved* mnp2_scale)
        ax[0].set_xticks([])
        ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['V2b', 'V1'],loc='upper right',fontsize='x-small') 
        ax[2].legend(['V2a_F', 'V2a_E'],loc='upper right',fontsize='x-small') 
        ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
        ax[3].set_xlabel('Time (ms)')
        ax[3].set_xticks(xticks)
        ax[3].set_xticklabels([f'{int(x)}' for x in xticks])
        ax[0].set_ylabel('Freq (Hz)')
        ax[1].set_ylabel('Freq (Hz)')
        ax[2].set_ylabel('Freq (Hz)')
        ax[3].set_ylabel('Freq (Hz)')
        ax[0].set_title(f'{label} - Average Spike Rate')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_spike_rate_rg_mnp.png',bbox_inches="tight")
        
        fig, ax = plt.subplots(4,sharex='all',figsize=(18, 12))    
        ax[0].plot(t, v0c1_convolved* v0c1_scale)
        ax[0].plot(t, v0c2_convolved* v0c2_scale)
        ax[1].plot(t, v1a1_convolved* v1a1_scale)
        ax[1].plot(t, v1a2_convolved* v1a2_scale)
        ax[2].plot(t, rc1_convolved* rc1_scale)
        ax[2].plot(t, rc2_convolved* rc2_scale)
        ax[3].plot(t, mnp1_convolved* mnp1_scale)
        ax[3].plot(t, mnp2_convolved* mnp2_scale)
        ax[0].set_xticks([])
        ax[0].legend(['V0c_F', 'V0c_E'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['1a_F', '1a_E'],loc='upper right',fontsize='x-small') 
        ax[2].legend(['RC_F', 'RC_E'],loc='upper right',fontsize='x-small') 
        ax[3].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
        ax[3].set_xlabel('Time (ms)')
        ax[3].set_xticks(xticks)
        ax[3].set_xticklabels([f'{int(x)}' for x in xticks])
        ax[0].set_ylabel('Freq (Hz)')
        ax[1].set_ylabel('Freq (Hz)')
        ax[2].set_ylabel('Freq (Hz)')
        ax[3].set_ylabel('Freq (Hz)')
        ax[0].set_title(f'{label} - Average Spike Rate')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_spike_rate_interneurons.png',bbox_inches="tight")

        # v1 overlap plot 
        if nn.args['overlap_plot']: 

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.2])

            # --- Top plot (V1) ---
            ax_v1_overlap = fig.add_subplot(gs[0, 0])

            ax_v1_overlap.plot(
                t,
                v1_convolved * float(v1_scale),
                color="tab:blue",
                label="V1 ipsi"
            )

            ax_v1_overlap.plot(
                t,
                v1_contra_convolved * float(v1_scale_contra),
                linestyle="--",
                color="tab:red",
                label="V1 contra"
            )

            ax_v1_overlap.set_title("V1 Ipsilateral and Contralateral Overlap Plot")
            ax_v1_overlap.set_ylabel("Frequency (Hz)")
            ax_v1_overlap.set_xlabel("Time (ms)")
            ax_v1_overlap.legend(loc='upper right', fontsize=24, ncol=2)

            # --- Bottom plot (MNP) ---
            ax_mnp_left_right = fig.add_subplot(gs[1, 0])

            ax_mnp_left_right.plot(
                t,
                mnp1_convolved * mnp1_scale,
                color="tab:blue",
                label="MNP Flx ipsi"
            )

            ax_mnp_left_right.plot(
                t,
                contra_mnp1_convolved * contra_mnp1_scale,
                linestyle="--",
                color="tab:red",
                label="MNP Flx contra"
            )

            ax_mnp_left_right.set_title("P112 Flexor-Flexor MNP Traces Overlap")
            ax_mnp_left_right.set_ylabel("Frequency (Hz)")
            ax_mnp_left_right.set_xlabel("Time (ms)")
            ax_mnp_left_right.legend(loc='upper right', fontsize=24, ncol=2)

            plt.tight_layout()

            if nn.args['save_as_svg'] and nn.args['save_results']:
                plt.savefig(nn.pathFigures + f'/{label}_V1_overlap.svg',
                            bbox_inches="tight", transparent=True)
            elif nn.args['save_results']:
                plt.savefig(nn.pathFigures + f'/{label}_V1_overlap.png',
                            dpi=300, bbox_inches="tight")

            plt.close()
            
        # ==================== V0D MNP OVERLAP PLOT =================================
        if nn.args['overlap_plot'] and nn.args['low_locomotion_v0d_left'] and nn.args['low_locomotion_v0d_right']:

            # Ablation Plot Params
            PLOT_ABLATION_WINDOWS = False
            ABLATION_WINDOWS = [
                (0, 5000),
            ]
            ABLATION_COLOR = "lightcoral"
            ABLATION_ALPHA = 0.5
            ABLATION_LABEL = "V0D ABLATED"

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.2])

            # V0d Plot
            ax_v0d_left_right = fig.add_subplot(gs[0, 0])

            ax_v0d_left_right.plot(
                t,
                v0d_convolved * v0d_scale,
                color="tab:blue",
                label="V0d ipsi"
            )

            ax_v0d_left_right.plot(
                t,
                v0d_contra_convolved * v0d_contra_scale,
                linestyle="--",
                color="tab:red",
                label="V0d contra"
            )

        
            ax_v0d_left_right.set_title("V0d Ipsilateral and Contralateral")
            ax_v0d_left_right.set_ylabel("Frequency (Hz)")
            
            # Add ablation shading
            if PLOT_ABLATION_WINDOWS:
                popfunc.plot_ablation_regions(
                    ax_v0d_left_right,
                    ABLATION_WINDOWS,
                    color=ABLATION_COLOR,
                    alpha=ABLATION_ALPHA,
                    label=ABLATION_LABEL
                )
            
            ax_v0d_left_right.legend(fontsize=24, ncol=2)

            # ==========================
            # MNP Plot
            # ==========================
            ax_mnp_left_right = fig.add_subplot(gs[1, 0])

            ax_mnp_left_right.plot(
                t,
                mnp1_convolved * mnp1_scale,
                color="tab:blue",
                label="MNP Flx ipsi"
            )

            ax_mnp_left_right.plot(
                t,
                contra_mnp1_convolved * contra_mnp1_scale,
                linestyle="--",
                color="tab:red",
                label="MNP Flx contra"
            )

            ax_mnp_left_right.set_title("P112 Flexor-Flexor MNP Traces Overlap")
            ax_mnp_left_right.set_ylabel("Frequency (Hz)")
            ax_mnp_left_right.set_xlabel("Time (ms)")
        
            if PLOT_ABLATION_WINDOWS:
                popfunc.plot_ablation_regions(
                    ax_mnp_left_right,
                    ABLATION_WINDOWS,
                    color=ABLATION_COLOR,
                    alpha=ABLATION_ALPHA,
                    label=ABLATION_LABEL
                    )
            
            ax_mnp_left_right.legend(loc='upper right', fontsize=24, ncol=2)

            # Layout
            plt.tight_layout()

            if nn.args['save_as_svg'] and nn.args['save_results']:
                plt.savefig(nn.pathFigures + f'/{label}_V0d_MNP_overlap.svg',
                            bbox_inches="tight", transparent=True)
            elif nn.args['save_results']:
                plt.savefig(nn.pathFigures + f'/{label}_V0d_MNP_overlap.png',
                            dpi=300, bbox_inches="tight")

            plt.close()


        # ======================= V3 MNP FLEXOR  OVERLAP PLOT ========================================
        if nn.args['overlap_plot'] and nn.args['low_locomotion_v3_right'] and nn.args['low_locomotion_v3_left']:

            print("Running the V3 Flexor Overlap Plot")

            # Ablation Plot Params
            PLOT_ABLATION_WINDOWS = False

            ABLATION_WINDOWS = [
                (0, 5000)
                # (15000, 17000),
            ]

            ABLATION_COLOR = "lightcoral"
            ABLATION_ALPHA = 0.5
            ABLATION_LABEL = "V3 ABLATED"

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.2])

            # V3 Plot
            ax_v3_left_right = fig.add_subplot(gs[0, 0])

            ax_v3_left_right.plot(
                t,
                v3_F_convolved * v3_F_scale,
                color="tab:blue",
                label="V3 FLX ipsi"
            )

            ax_v3_left_right.plot(
                t,
                v3_F_contra_convolved * v3_F_contra_scale,
                linestyle="--",
                color="tab:red",
                label="V3 FLX contra"
            )

            # Add ablation shading
            if PLOT_ABLATION_WINDOWS:
                popfunc.plot_ablation_regions(
                    ax_v3_left_right,
                    ABLATION_WINDOWS,
                    color=ABLATION_COLOR,
                    alpha=ABLATION_ALPHA,
                    label=ABLATION_LABEL
                )

            ax_v3_left_right.set_title("V3 Ipsilateral and Contralateral")
            ax_v3_left_right.set_ylabel("Frequency (Hz)")
            ax_v3_left_right.legend(fontsize=24, ncol=2)

            # ==========================
            # MNP Plot
            # ==========================

            ax_mnp_left_right = fig.add_subplot(gs[1, 0])

            ax_mnp_left_right.plot(
                t,
                mnp1_convolved * mnp1_scale,
                color="tab:blue",
                label="MNP Flx ipsi"
            )

            ax_mnp_left_right.plot(
                t,
                contra_mnp1_convolved * contra_mnp1_scale,
                linestyle="--",
                color="tab:red",
                label="MNP Flx contra"
            )

            # Add ablation shading
            if PLOT_ABLATION_WINDOWS:
                popfunc.plot_ablation_regions(
                    ax_mnp_left_right,
                    ABLATION_WINDOWS,
                    color=ABLATION_COLOR,
                    alpha=ABLATION_ALPHA,
                    label=ABLATION_LABEL
                )

            ax_mnp_left_right.set_ylabel("Frequency (Hz)")
            ax_mnp_left_right.set_xlabel("Time (ms)")
            ax_mnp_left_right.legend(fontsize=24, ncol=2)

             # Layout
            plt.tight_layout()

            if nn.args['save_as_svg'] and nn.args['save_results']:
                plt.savefig(nn.pathFigures + f'/{label}_V3_Flx_MNP_overlap.svg',
                            bbox_inches="tight", transparent=True)
            elif nn.args['save_results']:
                plt.savefig(nn.pathFigures + f'/{label}_V3_Flx_MNP_overlap.png',
                            dpi=300, bbox_inches="tight")

            plt.close()

        # ============================== RG V0D MNP PLOT ===========================
        if nn.args['low_locomotion_v0d_right'] and nn.args['low_locomotion_v0d_left']:
  
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.2])  # RG, V0d, MNP

            # --- Row 1: RG ---
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small", ncol=2)

    
            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle='--', alpha=0.6)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle='--', alpha=0.6)
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small", ncol=2)


            # --- Row 2: V0d (split into ipsilateral + contralateral) ---
            ax_v0d_left = fig.add_subplot(gs[1, 0])
            ax_v0d_right = fig.add_subplot(gs[1, 1])

            ax_v0d_left.plot(t, v0d_convolved * v0d_scale, color="tab:blue")
            ax_v0d_left.set_title("V0d Ipsilateral (Left Inhibitory Output)")
            ax_v0d_left.set_ylabel("Freq (Hz)")

            ax_v0d_right.plot(t, v0d_contra_convolved * v0d_contra_scale, color="tab:red")
            ax_v0d_right.set_title("V0d Contralateral (Cross-midline Inhibition)")

            # --- Row 3: MNP ---
            ax_mnp_left = fig.add_subplot(gs[2, 0])
            ax_mnp_right = fig.add_subplot(gs[2, 1])

            ax_mnp_left.plot(t, mnp1_convolved * mnp1_scale)
            ax_mnp_left.plot(t, mnp2_convolved * mnp2_scale)

            ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small")

            
            ax_mnp_right.plot(t, contra_mnp1_convolved * contra_mnp1_scale, linestyle='--', alpha=0.6)
            ax_mnp_right.plot(t, contra_mnp2_convolved * contra_mnp2_scale, linestyle='--', alpha=0.6)
            ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small")

            # Shared X labels bottom row only
            ax_mnp_left.set_xlabel("Time (ms)")
            ax_mnp_right.set_xlabel("Time (ms)")

            plt.tight_layout()

            if nn.args['save_results']:
                plt.savefig(nn.pathFigures + '/' + f"{label}_RG_V0d_split_MNP_combined.png",
                            dpi=300, bbox_inches="tight")

            plt.close()
        
        
        # ============================== RG V0D MNP PLOT ===========================
        if nn.args['low_locomotion_v3_left'] and nn.args['low_locomotion_v3_right']:
  
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.2])  # RG, V0d, MNP

            # --- Row 2: V0d (split into ipsilateral + contralateral) ---
            ax_v3_left = fig.add_subplot(gs[0, 0])
            ax_v3_right = fig.add_subplot(gs[0, 1])

            ax_v3_left.plot(t, v3_F_convolved * v3_F_scale, color="tab:blue")
            ax_v3_left.plot(t, v3_E_convolved * v3_E_scale, color="tab:red")
            ax_v3_left.legend(["V3 Ipsi FLEX", "V3 Ipsi EXT"], fontsize="xx-small")
            ax_v3_left.set_title("V3 Trace IPSILATERAL (Flex + Ext)")
            ax_v3_left.set_ylabel("Freq (Hz)")
        
            ax_v3_right.plot(t, v3_F_contra_convolved * v3_F_contra_scale, color="tab:blue")
            ax_v3_right.plot(t, v3_E_contra_convolved * v3_E_contra_scale, color="tab:red")
            ax_v3_right.legend(["V3 Ipsi FLEX", "V3 Ipsi EXT"], fontsize="xx-small")
            ax_v3_right.set_title("V3 Trace CONTRALATERAL (Flex + Ext)")
            ax_v3_right.set_ylabel("Freq (Hz)")

            # --- Row 3: MNP ---
            ax_mnp_left = fig.add_subplot(gs[1, 0])
            ax_mnp_right = fig.add_subplot(gs[1, 1])

            ax_mnp_left.plot(t, mnp1_convolved * mnp1_scale)
            ax_mnp_left.plot(t, mnp2_convolved * mnp2_scale)

            ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small", ncol=2)
            ax_mnp_right.plot(t, contra_mnp1_convolved * contra_mnp1_scale, linestyle='--', alpha=0.6)
            ax_mnp_right.plot(t, contra_mnp2_convolved * contra_mnp2_scale, linestyle='--', alpha=0.6)
            ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small", ncol=2)

            # Shared X labels bottom row only
            ax_mnp_left.set_xlabel("Time (ms)")
            ax_mnp_right.set_xlabel("Time (ms)")

            plt.tight_layout()

            if nn.args['save_results']:
                plt.savefig(nn.pathFigures + '/' + f"{label}_RG_V3_split_MNP_combined.png",
                            dpi=300, bbox_inches="tight")

            plt.close()
        
        if (ramp_type in ("online_ramp_drive", "online_ramp_weight") and scale_log is not None and nn.args["low_locomotion_v0d_left"] == 1 and nn.args["low_locomotion_v0d_right"] == 1):
            
            print("[INFO] V0D Online Ramp Plotting In Progress...")

            # --------------------------
            # Extract ramp log
            # --------------------------
            ramp_t = np.asarray(scale_log.get("time", []), dtype=float)

            if ramp_type == "online_ramp_drive":
                ramp_y = np.asarray(scale_log.get("drive", scale_log.get("mean_drive", [])), dtype=float)
                ramp_name = "Drive (I_e)"
            else:
                ramp_y = np.asarray(scale_log.get("weight", scale_log.get("mean_weight", [])), dtype=float)
                ramp_name = "Weight"

            # Safety
            if ramp_t.size == 0 or ramp_y.size == 0 or ramp_t.size != ramp_y.size:
                print("[WARN] Ramp log invalid; skipping ONLINE ramp plot.")
            else:
                # Sort ramp log
                order = np.argsort(ramp_t)
                ramp_t = ramp_t[order]
                ramp_y = ramp_y[order]

                # Ensure t is array
                t = np.asarray(t, dtype=float)

                # Interpolate ramp onto convolved timebase
                ramp_interp = np.interp(t, ramp_t, ramp_y, left=ramp_y[0], right=ramp_y[-1])

                # ==========================
                # FIGURE LAYOUT
                # ==========================
                fig = plt.figure(figsize=(16, 13), constrained_layout=True)
                gs = gridspec.GridSpec(
                    4, 1,
                    height_ratios=[1.2, 1.0, 1.2, 0.7],
                    figure=fig
                )

                fig.suptitle(
                    f"{label} | ONLINE RAMP | {ramp_name}",
                    fontsize=14
                )

                # --------------------------
                # ROW 1: RG LEFT (ipsi)
                # --------------------------
                ax_rg_left = fig.add_subplot(gs[0, 0])
                ax_rg_left.plot(t, rg1_convolved * rg1_scale)
                ax_rg_left.plot(t, rg2_convolved * rg2_scale)
                ax_rg_left.set_title("RG LEFT (ipsilateral)")
                ax_rg_left.legend(["RG_F", "RG_E"], fontsize="small")

                # --------------------------
                # ROW 2: V0D LEFT
                # --------------------------
                ax_v0d_left = fig.add_subplot(gs[1, 0], sharex=ax_rg_left)
                v0d_y = v0d_convolved * v0d_scale
                ax_v0d_left.plot(t, v0d_y)
                ax_v0d_left.set_title("V0d LEFT (inhibitory output)")
                ax_v0d_left.set_ylabel("Freq (Hz)")

                ymax = np.nanmax(v0d_y)
                if np.isfinite(ymax) and ymax > 0:
                    ax_v0d_left.set_ylim(0, ymax * 1.1)

                # --------------------------
                # ROW 3: RG RIGHT (contra)
                # --------------------------
                ax_rg_right = fig.add_subplot(gs[2, 0], sharex=ax_rg_left)
                ax_rg_right.plot(
                    t,
                    contra_rg1_convolved * contra_rg1_scale,
                    linestyle="--",
                    alpha=0.7,
                    label="RG_F contra"
                )
                ax_rg_right.plot(
                    t,
                    contra_rg2_convolved * contra_rg2_scale,
                    linestyle="--",
                    alpha=0.7,
                    label="RG_E contra"
                )
                ax_rg_right.set_title("RG RIGHT (contralateral)")
                ax_rg_right.legend(fontsize="small")

                # --------------------------
                # ROW 4: RAMP VS TIME
                # --------------------------
                ax_ramp = fig.add_subplot(gs[3, 0], sharex=ax_rg_left)
                ax_ramp.plot(t, ramp_interp)
                ax_ramp.set_title(f"{ramp_name} (applied ramp)")
                ax_ramp.set_xlabel("Time (ms)")
                ax_ramp.set_ylabel(ramp_name)

                # Shared x-limits
                xmin, xmax = float(t[0]), float(t[-1])
                for ax in (ax_rg_left, ax_v0d_left, ax_rg_right, ax_ramp):
                    ax.set_xlim(xmin, xmax)

                # Ramp y-lims with padding
                y_min, y_max = np.nanmin(ramp_interp), np.nanmax(ramp_interp)
                pad = 0.05 * (y_max - y_min + 1e-12)
                ax_ramp.set_ylim(y_min - pad, y_max + pad)
                ax_ramp.grid(True, axis="y", alpha=0.3)

                # --------------------------
                # SAVE
                # --------------------------
                if nn.args.get("save_results", 0):
                    fname = f"{label}_ONLINE_RAMP_V0D_{'DRIVE' if ramp_type=='online_ramp_drive' else 'WEIGHT'}.png"
                    plt.savefig(
                        nn.pathFigures + "/" + fname,
                        dpi=300,
                        bbox_inches="tight"
                    )
                    print(f"[INFO] Saved -> {fname}")

                plt.close(fig)

        else:
            ramp_t = None 
            ramp_w = None

         # ========================================== V0D WEIGHT STEPWISE PLOTTING ====================================================================  
        if (ramp_type == "online_stepwise_weight" and nn.stepwise_weight_experiment == 1 and scale_log is not None and nn.args["low_locomotion_v0d_left"] and nn.args["low_locomotion_v0d_right"]):

                print("[INFO] V0D Online Weight Stepwise Plotting In Progress")
                matplotlib.use("Agg")  # headless-safe

                # ---- 1) Load WEIGHT step log ----
                ramp_t = np.asarray(scale_log.get("time", []), dtype=float)
                ramp_w = np.asarray(scale_log.get("weight", []), dtype=float)

                if ramp_t.size == 0 or ramp_w.size == 0:
                    print("[WARN] Missing scale_log['time'] or scale_log['weight']; skipping plot.")
                elif ramp_t.size != ramp_w.size:
                    print(f"[WARN] scale_log time/weight length mismatch: {ramp_t.size} vs {ramp_w.size}; skipping plot.")
                else:
                    # sort logs
                    order = np.argsort(ramp_t)
                    ramp_t = ramp_t[order]
                    ramp_w = ramp_w[order]

                    # ---- 2) Canonical signal time axis ----
                    t_sig = np.asarray(convolved_time, dtype=float)
                    if t_sig.size == 0:
                        print("[WARN] convolved_time empty; skipping plot.")
                    else:
                        sig_order = np.argsort(t_sig)
                        t_sig = t_sig[sig_order]

                        # ---- 3) Put ALL signals on the SAME time base ----
                        def _as_sorted(x):
                            x = np.asarray(x, dtype=float)
                            if x.size == t_sig.size:
                                return x[sig_order]
                            return x  # fallback, but ideally sizes match

                        rgF_ipsi   = _as_sorted(rg1_convolved) * float(rg1_scale)
                        rgE_ipsi   = _as_sorted(rg2_convolved) * float(rg2_scale)
                        rgF_contra = _as_sorted(contra_rg1_convolved) * float(contra_rg1_scale)
                        rgE_contra = _as_sorted(contra_rg2_convolved) * float(contra_rg2_scale)

                        v0d_ipsi   = _as_sorted(v0d_convolved) * float(v0d_scale)
                        # if you have contralateral V0d convolved:
                        v0d_contra = _as_sorted(v0d_contra_convolved) * float(v0d_contra_scale) if "v0d_contra_convolved" in locals() else None

                        # ---- 4) Interpolate WEIGHT log onto signal time base ----
                        w_interp = np.interp(
                            t_sig,
                            ramp_t,
                            ramp_w,
                            left=ramp_w[0],
                            right=ramp_w[-1]
                        )

                        # ---- 5) Plot ----
                        fig = plt.figure(figsize=(16, 12))
                        gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 0.7])

                        fig.suptitle(f"{label} | ONLINE WEIGHT STEPWISE EXPERIMENT | {ramp_weight_name}", fontsize=14)

                        # --- Row 1: RG ---
                        ax_rg_left  = fig.add_subplot(gs[0, 0])
                        ax_rg_right = fig.add_subplot(gs[0, 1], sharex=ax_rg_left)

                        ax_rg_left.plot(t_sig, rgF_ipsi)
                        ax_rg_left.plot(t_sig, rgE_ipsi)
                        ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")
                        ax_rg_left.set_title("RG Ipsilateral")
                        ax_rg_left.set_ylabel("Freq (Hz)")

                        ax_rg_right.plot(t_sig, rgF_contra, linestyle="--", alpha=0.7)
                        ax_rg_right.plot(t_sig, rgE_contra, linestyle="--", alpha=0.7)
                        ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")
                        ax_rg_right.set_title("RG Contralateral")
                        ax_rg_right.set_ylabel("Freq (Hz)")

                        # --- Row 2: V0d ---
                        ax_v0d_left  = fig.add_subplot(gs[1, 0], sharex=ax_rg_left)
                        ax_v0d_right = fig.add_subplot(gs[1, 1], sharex=ax_rg_left)

                        ax_v0d_left.plot(t_sig, v0d_ipsi)
                        ax_v0d_left.set_title("V0d Ipsilateral")
                        ax_v0d_left.set_ylabel("Freq (Hz)")

                        if v0d_contra is not None:
                            ax_v0d_right.plot(t_sig, v0d_contra)
                            ax_v0d_right.set_title("V0d Contralateral")
                            ax_v0d_right.set_ylabel("Freq (Hz)")
                        else:
                            ax_v0d_right.axis("off")

                        # --- Row 3: WEIGHT (span both columns) ---
                        ax_w = fig.add_subplot(gs[2, :], sharex=ax_rg_left)
                        ax_w.plot(t_sig, w_interp)
                        ax_w.set_title(f"{ramp_weight_name} STEPWISE WEIGHT (aligned to convolved_time)")
                        ax_w.set_xlabel("Time (ms)")
                        ax_w.set_ylabel("Weight (applied)")

                        # Clamp x-lims explicitly (prevents union weirdness)
                        ax_rg_left.set_xlim(float(t_sig[0]), float(t_sig[-1]))

                        # Optional: nice y ticks
                        w_min = float(np.nanmin(w_interp))
                        w_max = float(np.nanmax(w_interp))
                        pad = 0.05 * (w_max - w_min + 1e-12)
                        ax_w.set_ylim(w_min - pad, w_max + pad)

                        n_ticks = 6
                        ticks = np.linspace(w_min, w_max, n_ticks)
                        ax_w.set_yticks(ticks)
                        ax_w.set_yticklabels([f"{v:.3f}" for v in ticks])
                        ax_w.grid(True, axis="y", alpha=0.3)

                        plt.tight_layout(rect=[0, 0, 1, 0.96])

                        if nn.args.get("save_results", 0):
                            plt.savefig(
                                nn.pathFigures + f"/{label}_ONLINE_STEPWISE_V0D_WEIGHT.png",
                                dpi=300,
                                bbox_inches="tight"
                            )
                        plt.close(fig)

        else:
            ramp_t = None
            ramp_w = None

            # ========================================== END OF WEIGHT V0D STEPWISE PLOTTING =================================================


        # =============================================== V3 WEIGHT STEPWISE EXPERIMENT ========================================================
        if (ramp_type == "online_stepwise_weight" and nn.stepwise_weight_experiment == 1 and scale_log is not None and nn.args["low_locomotion_v3_left"] and nn.args["low_locomotion_v3_right"]):

                print("[INFO] V3 Online Weight Stepwise Plotting In Progress")
                matplotlib.use("Agg")  # headless-safe

                # ---- 1) Load WEIGHT step log ----
                ramp_t = np.asarray(scale_log.get("time", []), dtype=float)
                ramp_w = np.asarray(scale_log.get("weight", []), dtype=float)

                if ramp_t.size == 0 or ramp_w.size == 0:
                    print("[WARN] Missing scale_log['time'] or scale_log['weight']; skipping plot.")
                elif ramp_t.size != ramp_w.size:
                    print(f"[WARN] scale_log time/weight length mismatch: {ramp_t.size} vs {ramp_w.size}; skipping plot.")
                else:
                    # sort logs
                    order = np.argsort(ramp_t)
                    ramp_t = ramp_t[order]
                    ramp_w = ramp_w[order]

                    # ---- 2) Canonical signal time axis ----
                    t_sig = np.asarray(convolved_time, dtype=float)
                    if t_sig.size == 0:
                        print("[WARN] convolved_time empty; skipping plot.")
                    else:
                        sig_order = np.argsort(t_sig)
                        t_sig = t_sig[sig_order]

                        # ---- 3) Put ALL signals on the SAME time base ----
                        def _as_sorted(x):
                            x = np.asarray(x, dtype=float)
                            if x.size == t_sig.size:
                                return x[sig_order]
                            return x  # fallback, but ideally sizes match

                        mnp1_ipsi   = _as_sorted(mnp1_convolved) * float(mnp1_scale)
                        mnp2_ipsi   = _as_sorted(mnp2_convolved) * float(mnp2_scale)
                        mnp1_contra = _as_sorted(contra_mnp1_convolved) * float(contra_mnp1_scale)
                        mnp2_contra = _as_sorted(contra_mnp2_convolved) * float(contra_mnp2_scale)

                        v3_f_ipsi = _as_sorted(v3_F_convolved) * float(v3_F_scale)
                        v3_e_ipsi   = _as_sorted(v3_E_convolved) * float(v3_E_scale)

                        v3_F_contra = _as_sorted(v3_F_contra_convolved) * float(v3_F_contra_scale)
                        v3_E_contra = _as_sorted(v3_E_contra_convolved) * float(v3_E_contra_scale)


                        # ---- 4) Interpolate WEIGHT log onto signal time base ----
                        w_interp = np.interp(
                            t_sig,
                            ramp_t,
                            ramp_w,
                            left=ramp_w[0],
                            right=ramp_w[-1]
                        )

                        # ---- 5) Plot ----
                        fig = plt.figure(figsize=(16, 12))
                        gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 0.7])

                        fig.suptitle(f"{label} | ONLINE WEIGHT STEPWISE EXPERIMENT | {ramp_weight_name}", fontsize=14)

                        # --- Row 1: mnp ---
                        ax_mnp_left  = fig.add_subplot(gs[0, 0])
                        ax_mnp_right = fig.add_subplot(gs[0, 1], sharex=ax_mnp_left)

                        ax_mnp_left.plot(t_sig, mnp1_ipsi)
                        ax_mnp_left.plot(t_sig, mnp2_ipsi)
                        ax_mnp_left.legend(["MNP FLX ISPI", "MNP EXT IPSI"], fontsize="xx-small", ncol=2)
                        ax_mnp_left.set_title("MNP Ipsilateral")
                        ax_mnp_left.set_ylabel("Freq (Hz)")

                        ax_mnp_right.plot(t_sig, mnp1_contra, linestyle="--", alpha=0.7)
                        ax_mnp_right.plot(t_sig, mnp2_contra, linestyle="--", alpha=0.7)
                        ax_mnp_right.legend(["MNP FLX CONTRA", "MNP EXT CONTRA"], fontsize="xx-small", ncol=2)
                        ax_mnp_right.set_title("MNP Contralateral")
                        ax_mnp_right.set_ylabel("Freq (Hz)")

                        # --- Row 2: V3s ---    
                        ax_v3_left  = fig.add_subplot(gs[1, 0], )
                        ax_v3_right = fig.add_subplot(gs[1, 1], sharex=ax_v3_left)

                        ax_v3_left.plot(t_sig, v3_f_ipsi, alpha=0.7)
                        ax_v3_left.plot(t_sig, v3_e_ipsi, linestyle="--", alpha=1.0)
                        ax_mnp_right.legend(["V3 FLX ipsi", "V3 EXT ipsi"], fontsize="xx-small", ncol=2)
                        ax_v3_left.set_title("V3 Ipsilateral (Flexor + Extensor)")
                        ax_v3_left.set_ylabel("Freq (Hz)")

                        ax_v3_right.plot(t_sig, v3_F_contra, alpha=0.7)
                        ax_v3_right.plot(t_sig, v3_E_contra,  linestyle="--", alpha=0.7)
                        ax_v3_right.legend(["V3 FLX contra", "V3 EXT contra"], fontsize="xx-small", ncol=2)
                        ax_v3_right.set_title("V3 Contralateral (Flexor + Extensor)")
                        ax_v3_right.set_ylabel("Freq (Hz)")

                        # --- Row 3: WEIGHT (span both columns) ---
                        ax_w = fig.add_subplot(gs[2, :])
                        ax_w.plot(t_sig, w_interp)
                        ax_w.set_title(f"{ramp_weight_name} STEPWISE WEIGHT (aligned to convolved_time)")
                        ax_w.set_xlabel("Time (ms)")
                        ax_w.set_ylabel("Weight (applied)")

                        # Clamp x-lims explicitly (prevents union weirdness)
                        ax_w.set_xlim(float(t_sig[0]), float(t_sig[-1]))

                        # Optional: nice y ticks
                        w_min = float(np.nanmin(w_interp))
                        w_max = float(np.nanmax(w_interp))
                        pad = 0.05 * (w_max - w_min + 1e-12)
                        ax_w.set_ylim(w_min - pad, w_max + pad)

                        n_ticks = 6
                        ticks = np.linspace(w_min, w_max, n_ticks)
                        ax_w.set_yticks(ticks)
                        ax_w.set_yticklabels([f"{v:.3f}" for v in ticks])
                        ax_w.grid(True, axis="y", alpha=0.3)

                        plt.tight_layout(rect=[0, 0, 1, 0.96])

                        if nn.args.get("save_results", 0):
                            plt.savefig(
                                nn.pathFigures + f"/{label}_ONLINE_STEPWISE_V3_WEIGHT.png",
                                dpi=300,
                                bbox_inches="tight"
                            )
                        plt.close(fig)

        else:
            ramp_t = None
            ramp_w = None


        # ========================================== END OF WEIGHT V0D STEPWISE PLOTTING =================================================


        # ========================================== V3 STEPWISE DRIVE PLOTTING ===========================================================
        if (ramp_type == "online_stepwise_drive" and nn.stepwise_drive_experiment == 1 and scale_log is not None and nn.args["low_locomotion_v3_left"] and nn.args["low_locomotion_v3_right"]):

            print("[INFO] V3 Online Drive Stepwise Plotting In Progress")
            matplotlib.use("Agg")  # headless-safe

            ramp_t = np.asarray(scale_log.get("time", []), dtype=float)
            ramp_drive = np.asarray(scale_log.get("drive", []), dtype=float)

            if ramp_t.size == 0 or ramp_drive.size == 0:
                print("[WARN] Missing scale_log['time'] or scale_log['drive']; skipping plot.")
            elif ramp_t.size != ramp_drive.size:
                print(f"[WARN] scale_log time/drive length mismatch: {ramp_t.size} vs {ramp_drive.size}; skipping plot.")
            else:
                # sort logs
                order = np.argsort(ramp_t)
                ramp_t = ramp_t[order]
                ramp_drive = ramp_drive[order]

                # IMPORTANT: use convolved_time as the canonical plot axis for signals
                t_sig = np.asarray(convolved_time, dtype=float)

                # make sure t_sig is sorted
                if t_sig.size == 0:
                    print("[WARN] convolved_time empty; skipping plot.")
                else:
                    sig_order = np.argsort(t_sig)
                    t_sig = t_sig[sig_order]

                    # ---------------------------
                    # 2) Put ALL signals on the SAME time base (t_sig)
                    #    (This is what fixes the ~500ms cutoff/mismatch)
                    # ---------------------------
                    def _as_sorted(x):
                        x = np.asarray(x, dtype=float)
                        return x[sig_order] if x.size == t_sig.size else x

                   
                    mnp1_ipsi   = _as_sorted(mnp1_convolved) * float(mnp1_scale)
                    mnp2_ipsi   = _as_sorted(mnp2_convolved) * float(mnp2_scale)
                    mnp1_contra = _as_sorted(contra_mnp1_convolved) * float(contra_mnp1_scale)
                    mnp2_contra = _as_sorted(contra_mnp2_convolved) * float(contra_mnp2_scale)

                    v3_f_ipsi = _as_sorted(v3_E_convolved) * float(v3_F_scale)
                    v3_e_ipsi = _as_sorted(v3_E_convolved) * float(v3_F_scale)
                    v3_f_contra = _as_sorted(v3_F_contra_convolved) * float(v3_F_contra_scale)
                    v3_e_contra = _as_sorted(v3_E_contra_convolved) * float(v3_F_contra_scale)


                    # Interpolate DRIVE log onto the signal time base for clean alignment
                    drive_interp = np.interp(
                        t_sig,
                        ramp_t,
                        ramp_drive,
                        left=ramp_drive[0],
                        right=ramp_drive[-1]
                    )


                    # Plotting code. 
                    fig = plt.figure(figsize=(16, 12))
                    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 0.7])

                    fig.suptitle(f"{label} | ONLINE DRIVE STEPWISE EXPERIMENT | {ramp_weight_name}", fontsize=14)

                    # --- Row 1: MNP ---
                    ax_mnp_left  = fig.add_subplot(gs[0, 0])
                    ax_mnp_right = fig.add_subplot(gs[0, 1], sharex=ax_mnp_left)

                    ax_mnp_left.plot(t_sig, mnp1_ipsi)
                    ax_mnp_left.plot(t_sig, mnp2_ipsi)
                    ax_mnp_left.legend(["MNP FLX ipsi", "MNP EXT ipsi"], fontsize="xx-small", ncol=2)
                    ax_mnp_left.set_title("MNP IPSILATERAL")
                    ax_mnp_left.set_ylabel("Freq (Hz)")

                    ax_mnp_right.plot(t_sig, mnp1_contra, alpha=0.7)
                    ax_mnp_right.plot(t_sig, mnp2_contra, linestyle="--", alpha=0.7)
                    ax_mnp_right.legend(["MNP FLX contra", "MNP EXT contra"], fontsize="xx-small", ncol=2)
                    ax_mnp_right.set_title("MNP CONTRALATERAL")
                    ax_mnp_right.set_ylabel("Freq (Hz)")

                    # --- Row 2: V3 ---
                    ax_v3_left  = fig.add_subplot(gs[1, 0], )
                    ax_v3_right = fig.add_subplot(gs[1, 1], sharex=ax_v3_left)

                    ax_v3_left.plot(t_sig, v3_f_ipsi, alpha=0.7)
                    ax_v3_left.plot(t_sig, v3_e_ipsi, linestyle="--", alpha=1.0)
                    ax_mnp_right.legend(["V3 FLX ipsi", "V3 EXT ipsi"], fontsize="xx-small", ncol=2)
                    ax_v3_left.set_title("V3 Ipsilateral (Flexor + Extensor)")
                    ax_v3_left.set_ylabel("Freq (Hz)")

                    ax_v3_right.plot(t_sig, v3_f_contra, alpha=0.7)
                    ax_v3_right.plot(t_sig, v3_e_contra,  linestyle="--", alpha=0.7)
                    ax_v3_right.legend(["V3 FLX contra", "V3 EXT contra"], fontsize="xx-small", ncol=2)
                    ax_v3_right.set_title("V3 Contralateral (Flexor + Extensor)")
                    ax_v3_right.set_ylabel("Freq (Hz)")

                    # --- Row 3: DRIVE (span both columns) ---
                    ax_drive = fig.add_subplot(gs[2, :], sharex=ax_mnp_left)
                    ax_drive.plot(t_sig, drive_interp)
                    ax_drive.set_title(f"{ramp_weight_name} STEPWISE DRIVE (aligned to convolved_time)")
                    ax_drive.set_xlabel("Time (ms)")
                    ax_drive.set_ylabel("Drive (applied)")

                    # Clamp x-lims explicitly (avoid autoscale oddities)
                    ax_mnp_left.set_xlim(float(t_sig[0]), float(t_sig[-1]))

                    # ---- Y-axis ticks & limits ----
                    w_min = float(np.nanmin(drive_interp))
                    w_max = float(np.nanmax(drive_interp))

                    # Add padding so it doesn't clip
                    pad = 0.05 * (w_max - w_min + 1e-12)
                    ax_drive.set_ylim(w_min - pad, w_max + pad)

                    # Choose a sensible number of ticks
                    n_ticks = 6  # 6–8 is usually clean
                    ticks = np.linspace(w_min, w_max, n_ticks)
                    ax_drive.set_yticks(ticks)
                    ax_drive.set_yticklabels([f"{v:.3f}" for v in ticks])

                    # Light horizontal grid for readability
                    ax_drive.grid(True, axis="y", alpha=0.3)

                    plt.tight_layout(rect=[0, 0, 1, 0.96])

                    # Minor grid to show ramp smoothly
                    ax_drive.grid(True, axis="y", alpha=0.3)

                    if nn.args.get("save_results", 0):
                        plt.savefig(
                            nn.pathFigures + f"/{label}_ONLINE_STEPWISE_V3_DRIVE.png",
                            dpi=300,
                            bbox_inches="tight"
                    )
                    plt.close(fig)

        else:
            ramp_t = None
            ramp_w = None
        # =================================================== END OF V3 STEPWISE PLOTTING =================================================



        # # ========================================== V0D DRIVE STEPWISE PLOTTING ================================================= 
        if (ramp_type == "online_stepwise_drive" and nn.stepwise_drive_experiment == 1 and scale_log is not None and nn.args["low_locomotion_v0d_left"] and nn.args["low_locomotion_v0d_right"]):

            print("[INFO] V0D Online Drive Stepwise Plotting In Progress")
            matplotlib.use("Agg")  # headless-safe

            ramp_t = np.asarray(scale_log.get("time", []), dtype=float)
            ramp_drive = np.asarray(scale_log.get("drive", []), dtype=float)

            if ramp_t.size == 0 or ramp_drive.size == 0:
                print("[WARN] Missing scale_log['time'] or scale_log['drive']; skipping plot.")
            elif ramp_t.size != ramp_drive.size:
                print(f"[WARN] scale_log time/drive length mismatch: {ramp_t.size} vs {ramp_drive.size}; skipping plot.")
            else:
                # sort logs
                order = np.argsort(ramp_t)
                ramp_t = ramp_t[order]
                ramp_drive = ramp_drive[order]

                # IMPORTANT: use convolved_time as the canonical plot axis for signals
                t_sig = np.asarray(convolved_time, dtype=float)

                # make sure t_sig is sorted
                if t_sig.size == 0:
                    print("[WARN] convolved_time empty; skipping plot.")
                else:
                    sig_order = np.argsort(t_sig)
                    t_sig = t_sig[sig_order]

                    # ---------------------------
                    # 2) Put ALL signals on the SAME time base (t_sig)
                    #    (This is what fixes the ~500ms cutoff/mismatch)
                    # ---------------------------
                    def _as_sorted(x):
                        x = np.asarray(x, dtype=float)
                        return x[sig_order] if x.size == t_sig.size else x

                    rgF_ipsi   = _as_sorted(rg1_convolved) * float(rg1_scale)
                    rgE_ipsi   = _as_sorted(rg2_convolved) * float(rg2_scale)
                    rgF_contra = _as_sorted(contra_rg1_convolved) * float(contra_rg1_scale)
                    rgE_contra = _as_sorted(contra_rg2_convolved) * float(contra_rg2_scale)

                    v0d_ipsi   = _as_sorted(v0d_convolved) * float(v0d_scale)
                    v0d_contra = _as_sorted(v0d_contra_convolved) * float(v0d_contra_scale)

                    # Interpolate DRIVE log onto the signal time base for clean alignment
                    drive_interp = np.interp(
                        t_sig,
                        ramp_t,
                        ramp_drive,
                        left=ramp_drive[0],
                        right=ramp_drive[-1]
                    )


                    # Plotting code. 
                    fig = plt.figure(figsize=(16, 12))
                    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 0.7])

                    fig.suptitle(f"{label} | ONLINE DRIVE STEPWISE EXPERIMENT | {ramp_weight_name}", fontsize=14)

                    # --- Row 1: RG ---
                    ax_rg_left  = fig.add_subplot(gs[0, 0])
                    ax_rg_right = fig.add_subplot(gs[0, 1], sharex=ax_rg_left)

                    ax_rg_left.plot(t_sig, rgF_ipsi)
                    ax_rg_left.plot(t_sig, rgE_ipsi)
                    ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small", ncol=2)
                    ax_rg_left.set_title("RG Ipsilateral")
                    ax_rg_left.set_ylabel("Freq (Hz)")

                    ax_rg_right.plot(t_sig, rgF_contra, linestyle="--", alpha=0.7)
                    ax_rg_right.plot(t_sig, rgE_contra, linestyle="--", alpha=0.7)
                    ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small", ncol=2)
                    ax_rg_right.set_title("RG Contralateral")
                    ax_rg_right.set_ylabel("Freq (Hz)")

                    # --- Row 2: V0d ---
                    ax_v0d_left  = fig.add_subplot(gs[1, 0], sharex=ax_rg_left)
                    ax_v0d_right = fig.add_subplot(gs[1, 1], sharex=ax_rg_left)

                    ax_v0d_left.plot(t_sig, v0d_ipsi)
                    ax_v0d_left.set_title("V0d Ipsilateral")
                    ax_v0d_left.set_ylabel("Freq (Hz)")

                    ax_v0d_right.plot(t_sig, v0d_contra)
                    ax_v0d_right.set_title("V0d Contralateral")
                    ax_v0d_right.set_ylabel("Freq (Hz)")

                    # --- Row 3: DRIVE (span both columns) ---
                    ax_drive = fig.add_subplot(gs[2, :], sharex=ax_rg_left)
                    ax_drive.plot(t_sig, drive_interp)
                    ax_drive.set_title(f"{ramp_weight_name} STEPWISE DRIVE (aligned to convolved_time)")
                    ax_drive.set_xlabel("Time (ms)")
                    ax_drive.set_ylabel("Drive (applied)")

                    # Clamp x-lims explicitly (avoid autoscale oddities)
                    ax_rg_left.set_xlim(float(t_sig[0]), float(t_sig[-1]))

                    # ---- Y-axis ticks & limits ----
                    w_min = float(np.nanmin(drive_interp))
                    w_max = float(np.nanmax(drive_interp))

                    # Add padding so it doesn't clip
                    pad = 0.05 * (w_max - w_min + 1e-12)
                    ax_drive.set_ylim(w_min - pad, w_max + pad)

                    # Choose a sensible number of ticks
                    n_ticks = 6  # 6–8 is usually clean
                    ticks = np.linspace(w_min, w_max, n_ticks)
                    ax_drive.set_yticks(ticks)
                    ax_drive.set_yticklabels([f"{v:.3f}" for v in ticks])

                    # Light horizontal grid for readability
                    ax_drive.grid(True, axis="y", alpha=0.3)

                    plt.tight_layout(rect=[0, 0, 1, 0.96])

                    # Minor grid to show ramp smoothly
                    ax_drive.grid(True, axis="y", alpha=0.3)

                    if nn.args.get("save_results", 0):
                        plt.savefig(
                            nn.pathFigures + f"/{label}_ONLINE_STEPWISE_V0D_DRIVE.png",
                            dpi=300,
                            bbox_inches="tight"
                        )
                    plt.close(fig)

        else:
            ramp_t = None
            ramp_w = None
            # # ========================================== END OF DRIVE STEPWISE PLOTTING ================================


        # PLOTTING V0D ASYMMETRIC DEGENERATION EXPERIMENTS 
        if nn.args['v0d_degeneration_plot'] and nn.args["low_locomotion_v0d_left"] and nn.args["low_locomotion_v0d_right"] and nn.asymmetric_onset == 1: 

            days_after_onset = nn.days
            print(f"[INFO] Plotting V0d Symmetric Degeneration: {days_after_onset}")

            #  FIX: define t_sig and sig_order BEFORE _as_sorted and the signals ──
            t_sig = np.asarray(convolved_time, dtype=float)
            if t_sig.size == 0:
                print("[WARN] convolved_time empty; skipping plot.")
            else:
                sig_order = np.argsort(t_sig)
                t_sig = t_sig[sig_order]

                fig = plt.figure(figsize=(10, 15), constrained_layout=True)   # ← also moved here
                gs = gridspec.GridSpec(
                    4, 2,
                    height_ratios=[1.2, 1.2, 1.2, 0.6],
                    figure=fig
                )

                def _as_sorted(x):
                    x = np.asarray(x, dtype=float)
                    return x[sig_order] if x.size == t_sig.size else x

                rgF_ipsi   = _as_sorted(rg1_convolved)       * float(rg1_scale)
                rgE_ipsi   = _as_sorted(rg2_convolved)       * float(rg2_scale)
                rgF_contra = _as_sorted(contra_rg1_convolved) * float(contra_rg1_scale)
                rgE_contra = _as_sorted(contra_rg2_convolved) * float(contra_rg2_scale)
                v0d_ipsi   = _as_sorted(v0d_convolved)       * float(v0d_scale)
                v0d_contra = _as_sorted(v0d_contra_convolved) * float(v0d_contra_scale)
                v1_ipsi    = _as_sorted(v1_convolved)         * float(v1_scale)

            fig.suptitle(f"{label} | DEGENERATION EXPERIMENT | day {days_after_onset}", fontsize=25)
               
            # ---------- Row 1: V1 ----------
            ax_v1_deg = fig.add_subplot(gs[0, 0])
            ax_v1_deg.plot(t_sig, v1_ipsi)
            ax_v1_deg.set_title("V1 Left Hemicord (Ipsilateral)")
            ax_v1_deg.set_ylabel("Freq (Hz)")

            # ---------- Row 2: RG (ipsi + contra together) ----------
            ax_rg = fig.add_subplot(gs[1, 0])

            ax_rg.plot(t_sig, rgF_ipsi)
            ax_rg.plot(t_sig, rgE_ipsi)
            ax_rg.set_title("RG Activity (Ipsi)")
            ax_rg.legend(
                ["RG_F ipsi", "RG_E ipsi", "RG_F contra", "RG_E contra"],
                fontsize="xx-small"
            )

            # ---------- Row 3: V0d Left ----------
            ax_v0d_left = fig.add_subplot(gs[2, 0], sharex=ax_rg)

            ax_v0d_left.plot(t_sig, v0d_ipsi)
            ax_v0d_left.set_title("V0d LEFT")
            ax_v0d_left.set_ylabel("Freq (Hz)")

            ymax = np.nanmax(v0d_ipsi)
            if np.isfinite(ymax) and ymax > 0:
                ax_v0d_left.set_ylim(0, ymax * 1.1)

            # ---------- Row 4: RG Contra (signal-level) ----------
            ax_rg_contra = fig.add_subplot(gs[3, 0], sharex=ax_rg)

            ax_rg_contra.plot(t_sig, rgF_contra, linestyle="-", alpha=0.2)
            ax_rg_contra.plot(t_sig, rgE_contra, linestyle="-", alpha=0.2)
            ax_rg_contra.set_title("RG Contralateral (Signal)")
            ax_rg_contra.set_ylabel("Freq (Hz)")
            ax_rg_contra.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")

            # ---------- Save ----------
            if nn.args.get("save_results", 0):
                plt.savefig(
                    nn.pathFigures + f"/{label}_ASYMMETRIC_V0D_DEG.png",
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.close(fig)


        # PLOTTING V0D SYMMETRIC DEGENERATION EXPERIMENTS 
        if nn.args['v0d_degeneration_plot'] and nn.args["low_locomotion_v0d_left"] and nn.args["low_locomotion_v0d_right"] and nn.asymmetric_onset == 0:

            days_after_onset = nn.days
            print(f"[INFO] Plotting V0d Symmetric Degeneration: {days_after_onset}")

            gs = gridspec.GridSpec(
                4, 2,
                height_ratios=[1.2, 1.2, 1.2, 0.6],
                figure=fig
            )

            t_sig = np.asarray(convolved_time, dtype=float)
            if t_sig.size == 0:
                print("[WARN] convolved_time empty; skipping plot.")
                return  # or handle safely

            sig_order = np.argsort(t_sig)
            t_sig = t_sig[sig_order]

            def _as_sorted(x):
                x = np.asarray(x, dtype=float)
                return x[sig_order] if x.size == t_sig.size else x

            rgF_ipsi   = _as_sorted(rg1_convolved) * float(rg1_scale)
            rgE_ipsi   = _as_sorted(rg2_convolved) * float(rg2_scale)
            rgF_contra = _as_sorted(contra_rg1_convolved) * float(contra_rg1_scale)
            rgE_contra = _as_sorted(contra_rg2_convolved) * float(contra_rg2_scale)

            v0d_ipsi   = _as_sorted(v0d_convolved) * float(v0d_scale)
            v0d_contra = _as_sorted(v0d_contra_convolved) * float(v0d_contra_scale)
            v1_ipsi = _as_sorted(v1_convolved) * float(v1_scale)
            v1_contra = _as_sorted(v1_contra_convolved) * float(v1_scale_contra)
                   
            fig.suptitle(f"{label} | SYMMETRIC DEGENERATION EXPERIMENT | day {days_after_onset}", fontsize=25)
            
            # ---------- Row 0: V1 (ipsi/contra) ---------- we need to implement _L and _R here. 
            ax_v1_left_deg  = fig.add_subplot(gs[0, 0])
            ax_v1_right_deg = fig.add_subplot(gs[0, 1], sharex=ax_v1_left_deg)

            
            ax_v1_left_deg.plot(t_sig, v1_ipsi)
            ax_v1_left_deg.set_title("V1 Ipsilateral")
            ax_v1_left_deg.set_ylabel("Freq (Hz)")

            ax_v1_right_deg.plot(t_sig, v1_contra)
            ax_v1_right_deg.set_title("V1 Contralateral")

            # ---------- Row 1: RG ----------
            ax_rg_left  = fig.add_subplot(gs[1, 0], sharex=ax_v1_left_deg)
            ax_rg_right = fig.add_subplot(gs[1, 1], sharex=ax_v1_left_deg)

            ax_rg_left.plot(t_sig, rgF_ipsi)
            ax_rg_left.plot(t_sig, rgE_ipsi)
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")
            ax_rg_left.set_title("RG Ipsilateral")
            ax_rg_left.set_ylabel("Freq (Hz)")

            ax_rg_right.plot(t_sig, rgF_contra, linestyle="--", alpha=0.7)
            ax_rg_right.plot(t_sig, rgE_contra, linestyle="--", alpha=0.7)
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")
            ax_rg_right.set_title("RG Contralateral")

            # ---------- Row 2: V0d ----------
            ax_v0d_left  = fig.add_subplot(gs[2, 0], sharex=ax_v1_left_deg)
            ax_v0d_right = fig.add_subplot(gs[2, 1], sharex=ax_v1_left_deg)

            ax_v0d_left.plot(t_sig, v0d_ipsi)
            ax_v0d_left.set_title("V0d Ipsilateral")
            ax_v0d_left.set_ylabel("Freq (Hz)")

            ax_v0d_right.plot(t_sig, v0d_contra)
            ax_v0d_right.set_title("V0d Contralateral")

            # ---------- Clamp x-lims (avoid autoscale weirdness) ----------
            xmin = float(min(t_sig[0], t[0]))
            xmax = float(max(t_sig[-1], t[-1]))
            for ax in (ax_v1_left_deg, ax_v1_right_deg, ax_rg_left, ax_rg_right, ax_v0d_left, ax_v0d_right):
                ax.set_xlim(xmin, xmax)

            ax_v0d_left.set_xlabel("Time (ms)")
            ax_v0d_right.set_xlabel("Time (ms)")

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if nn.args.get("save_results", 0):
                plt.savefig(
                    nn.pathFigures + f"/{label}_SYMMETRIC_V0D_PLOT.png",
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.close(fig)

            
            
    if nn.spike_distribution_plot==1:
        #Count spikes per neuron
        # Define parameters and senders for each neuron group
        neuron_params = [
            (nn.flx_exc_bursting_count, senders_exc1), 
            (nn.flx_inh_bursting_count, senders_inh1), 
            (nn.flx_exc_tonic_count, senders_exc_tonic1), 
            (nn.flx_inh_tonic_count, senders_inh_tonic1),
            (nn.ext_exc_bursting_count, senders_exc2), 
            (nn.ext_inh_bursting_count, senders_inh2), 
            (nn.ext_exc_tonic_count, senders_exc_tonic2), 
            (nn.ext_inh_tonic_count, senders_inh_tonic2),
            (nn.v2a_tonic_pop_size, senders_exc_inter_tonic1), 
            (nn.v2a_tonic_pop_size, senders_exc_inter_tonic2),
            (nn.v0c_pop_size,senders_V0c_1),
            (nn.v0c_pop_size,senders_V0c_2),
            (nn.v1a_pop_size,senders_V1a_1),
            (nn.v1a_pop_size,senders_V1a_2),
            (nn.rc_pop_size,senders_rc_1),
            (nn.rc_pop_size,senders_rc_2),
            (nn.num_motor_neurons, senders_mnp1), 
            (nn.num_motor_neurons, senders_mnp2)
        ]

        # If RGs are connected, add inhibitory inter-neurons
        if nn.rgs_connected == 1:
            neuron_params.extend([
                (nn.num_inh_inter_tonic_v2b, senders_inh_inter_tonic1), 
                (nn.num_inh_inter_tonic_v1, senders_inh_inter_tonic2)
            ])

        # Initialize counters for spikes, sparse firing, and silent neurons
        all_indiv_spike_counts = []
        sparse_firing_count = 0
        silent_neuron_count = 0

        # Iterate through all neuron groups and compute spike data
        for param, senders in neuron_params:
            indiv_spikes, _, sparse_count, silent_count = popfunc.count_indiv_spikes(param, senders, avg_freq)
            all_indiv_spike_counts.extend(indiv_spikes)
            sparse_firing_count += sparse_count
            silent_neuron_count += silent_count

        # Calculate and print sparse firing statistics
        active_neuron_count = len(all_indiv_spike_counts) - silent_neuron_count
        if len(all_indiv_spike_counts) > 0:
            sparse_firing_percentage = round(sparse_firing_count * 100 / (len(all_indiv_spike_counts) - silent_neuron_count), 2)
            print('Active neuron count, sparsely firing count, % sparse firing:', active_neuron_count, sparse_firing_count, sparse_firing_percentage, '%')
        else:
            print("No active neurons found; all neurons are silent.")       
        
        spike_distribution = [all_indiv_spike_counts.count(i) for i in range(max(all_indiv_spike_counts))]
        '''
        pylab.figure()
        pylab.plot(spike_distribution[2:])
        pylab.xscale('log')
        pylab.xlabel('Total Spike Count')
        pylab.ylabel('Number of Neurons')
        pylab.title('Spike Distribution')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution.png',bbox_inches="tight")
        '''
    
    if nn.membrane_potential_plot==1:
        v_m1,t_m1 = popfunc.read_membrane_potential(mnp1.mm_motor,nn.num_motor_neurons,mnp1.neuron_to_sample)
        v_m2,t_m2 = popfunc.read_membrane_potential(mnp2.mm_motor,nn.num_motor_neurons,mnp2.neuron_to_sample)

        pylab.figure(figsize=(18, 12))
        pylab.subplot(211)
        pylab.plot(t_m1,v_m1)
        pylab.xlim(1000,1500)
        pylab.title('Individual Neuron Membrane Potential (Flx)')
        pylab.ylabel('Membrane potential (mV)')
        pylab.subplot(212)
        pylab.plot(t_m2,v_m2)
        pylab.xlim(1000,1500)
        pylab.title(f'{label} - Individual Neuron Membrane Potential (Ext)')
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Membrane potential (mV)')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_membrane_potential_mns.png',bbox_inches="tight")

    #Plot individual spikes
    if nn.raster_plot==1:
        pylab.figure(figsize=(18, 12))
        pylab.subplot(211)
        for i in range(nn.num_motor_neurons-1): 
            pylab.plot(spiketimes_mnp1[0][i],senders_mnp1[0][i],'.',label='Flx')
        pylab.xlim(1000,1500)
        pylab.title('Spike Output (Flx)')
        pylab.ylabel('Neuron ID')
        pylab.subplot(212)
        for i in range(nn.num_motor_neurons-1):
            pylab.plot(spiketimes_mnp2[0][i],senders_mnp2[0][i],'.',label='Ext')  
        pylab.xlim(1000,1500)
        pylab.title('Spike Output (Ext)')
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_raster_plot_mns.png',bbox_inches="tight")

    if nn.fb_rg_flx == 1:
        #Plot poisson generator spikes
        fig,ax = plt.subplots(figsize=(18, 12))
        for i in range(nn.num_pgs-1):
            if nn.num_pgs != 0: ax.plot(spiketimes_rg_flx_pg[0][i],senders_rg_flx_pg[0][i],'.')
        #ax.set_ylim(2,12)
        #ax.set_xlim(500,4000)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')
        plt.title('Poisson Spiking Input')    
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_poisson_spike_input.png',bbox_inches="tight")        

    if nn.args['save_results']:        

        np.savetxt(nn.pathFigures + f'/{label}_output_mnp1.csv',mnp1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_mnp2.csv',mnp2_convolved_scaled,delimiter=',')    

    if nn.args['save_results'] and nn.save_v0d_pops==1:

        if not nn.low_locomotion_v0d_left or not nn.low_locomotion_v0d_right:
            print("V0d population is not activated. Skipping file saving.")
            return

        np.savetxt(nn.pathFigures + f'/{label}_output_v0d.csv',v0d_convolved_scaled,delimiter=',')

    if nn.args['save_results'] and nn.save_rg_v1_pops==1: 

        v1_convolved_scaled = spike_bins_inh_inter2
        rg1_convolved_scaled = spike_bins_rg1
        rg2_convolved_scaled = spike_bins_rg2
 
        np.savetxt(nn.pathFigures + f'/{label}_output_rg1.csv',rg1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rg2.csv',rg2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v1.csv',v1_convolved_scaled,delimiter=',')  

    if nn.args['save_results'] and nn.save_v3_pops==1: 

        if not nn.low_locomotion_v3_left or not nn.low_locomotion_v3_right:
            print("V3 population is not activated. Skipping file saving.")
            return
              
        np.savetxt(nn.pathFigures + f'/{label}_output_V3_F.csv',v3_F_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_V3_E.csv',v3_E_convolved_scaled,delimiter=',')

    if nn.args['save_results'] and nn.save_all_pops==1:

        rg1_convolved_scaled = spike_bins_rg1
        rg2_convolved_scaled = spike_bins_rg2
        v2b_convolved_scaled = spike_bins_inh_inter1
        v1_convolved_scaled = spike_bins_inh_inter2
        v2a1_convolved_scaled = spike_bins_exc_inter1
        v2a2_convolved_scaled = spike_bins_exc_inter2
        v0c1_convolved_scaled = spike_bins_V0c_1
        v0c2_convolved_scaled = spike_bins_V0c_2
        v1a1_convolved_scaled = spike_bins_V1a_1
        v1a2_convolved_scaled = spike_bins_V1a_2
        rc1_convolved_scaled = spike_bins_rc_1
        rc2_convolved_scaled = spike_bins_rc_2

        np.savetxt(nn.pathFigures + f'/{label}_output_rg1.csv',rg1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rg2.csv',rg2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v2b.csv',v2b_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v1.csv',v1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v2a1.csv',v2a1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v2a2.csv',v2a2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v0c1.csv',v0c1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v0c2.csv',v0c2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_1a1.csv',v1a1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_1a2.csv',v1a2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rc1.csv',rc1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rc2.csv',rc2_convolved_scaled,delimiter=',')  

   
    if nn.args['save_results'] and nn.rate_coded_plot == 1 and nn.isf_output == 0:
        # Save population rate output
        np.savetxt(nn.pathFigures + f'/{label}_output_mnp1.csv',spike_bins_mnp1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_mnp2.csv',spike_bins_mnp2_true,delimiter=',')
        
    if nn.args['save_results'] and nn.save_all_pops==1 and nn.rate_coded_plot == 1 and nn.isf_output == 0:
        np.savetxt(nn.pathFigures + f'/{label}_output_rg1.csv',spike_bins_rg1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rg2.csv',spike_bins_rg2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v2b.csv',spike_bins_inh_inter1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v1.csv',spike_bins_inh_inter2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v2a1.csv',spike_bins_exc_inter1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v2a2.csv',spike_bins_exc_inter2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v0c1.csv',spike_bins_V0c_1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_v0c2.csv',spike_bins_V0c_2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_1a1.csv',spike_bins_V1a_1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_1a2.csv',spike_bins_V1a_2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rc1.csv',spike_bins_rc_1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}_output_rc2.csv',spike_bins_rc_2_true,delimiter=',')

     # Parameter for saving metric summary table. Useful in Normal sims for quant analysis 
    
    if nn.args['save_metric_summary_table']:

        existing_metrics = {
            
            # already computed in cpg_utils/cpg_data_utils:
            "rg_flx_isf_max": rg1_isf_max,
            "rg_ext_isf_max": rg2_isf_max,
            "mnp_flx_isf_max": mnp1_isf_max,
            "mnp_ext_isf_max": mnp2_isf_max,
        }

          # should probably run a loop to understand how this is working 
        populations = {
            "MNP_ipsilateral": {
                "signals": {"FLX": mnp1_convolved_scaled, "EXT": mnp2_convolved_scaled},
                "pairs": [("FLX", "EXT")]
            },
            "RG_ipsilateral": {
                "signals": {"RG_F": rg1_convolved_scaled, "RG_E": rg2_convolved_scaled},
                "pairs": [("RG_F", "RG_E")]
            },
            "MNP_contralateral": {
                "signals": {"FLX": contra_mnp1_convolved * contra_mnp1_scale,
                            "EXT": contra_mnp2_convolved * contra_mnp2_scale},
                "pairs": [("FLX", "EXT")]
            },
            "RG_contralateral": {
                "signals": {"RG_F": contra_rg1_convolved * contra_rg1_scale,
                            "RG_E": contra_rg2_convolved * contra_rg2_scale},
                "pairs": [("RG_F", "RG_E")]
            },
        }

        meta = {
            "ramp_weight_name": ramp_weight_name,
            "w_start": nn.w_start,
            "w_end": nn.w_end,
            "ramp_type": ramp_type,
        }

        out_csv = nn.pathFigures + f"/{label}_metrics_summary_table_CLEAN.csv"

        df_metrics = popfunc.export_metrics_table_clean(
            out_csv_path=out_csv,
            label=label,
            t_ms=t,
            populations=populations,
            existing=existing_metrics,
            meta=meta,
            thresh_frac=0.2,
            min_peak_distance_ms=200.0,
            round_ndp=4,
        )

        print("[INFO] Saved CLEAN metrics table:", out_csv)
        print(df_metrics.to_string(index=False))
        # -------------------------------------------------------