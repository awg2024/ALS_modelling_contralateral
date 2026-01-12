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
              V0v, V0v_contra, 
              V0d, V0d_contra, 
              label, 
              ramp_weight=None,           # optional
              ramp_weight_name=None,         # optional
              ramp_type=None):                   # optional 
    
    
    print(f"CPG UTILS {label} Started.")
    
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

    # --- DEBUG V0c recording ---
    # print("\n[DEBUG V0c]")
    # print("V0c_1 detector:", V0c_1.spike_detector)
    # print("V0c_2 detector:", V0c_2.spike_detector)
    # time.sleep(5)


    senders_V1a_1,spiketimes_V1a_1 = popfunc.read_spike_data(V1a_1.spike_detector)
    senders_V1a_2,spiketimes_V1a_2 = popfunc.read_spike_data(V1a_2.spike_detector)
    senders_rc_1,spiketimes_rc_1 = popfunc.read_spike_data(rc_1.spike_detector)
    senders_rc_2,spiketimes_rc_2 = popfunc.read_spike_data(rc_2.spike_detector)


    senders_V0d_contra, spiketimes_V0d_contra = popfunc.read_spike_data(V0d_contra.spike_detector)
    
    senders_V0d, spiketimes_V0d = popfunc.read_spike_data(V0d.spike_detector)
    senders_V0v, spiketimes_V0v = popfunc.read_spike_data(V0v.spike_detector)
    
    senders_V0v_contra, spiketimes_V0v_contra = popfunc.read_spike_data(V0v_contra.spike_detector)

    # Read spike data — contralateral RG and MNP

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

    if nn.args['low_locomotion_v0d_right'] and nn.args['low_locomotion_v0d_left']:

        popfunc.spike_report("RG FLX IPSILATERAL", senders_rc_1,       spiketimes_rc_1)
        popfunc.spike_report("V0d IPSILATERAL",    senders_V0d,        spiketimes_V0d)
        
        popfunc.spike_report("RG FLX CONTRALATERAL EXC BURST", senders_contra_rg_exc_burst1, spiketimes_contra_rg_exc_burst1)
        popfunc.spike_report("V0d CONTRALATERAL",  senders_V0d_contra, spiketimes_V0d_contra)

        popfunc.spike_report("IPSILATERAL MNP FLX", senders_mnp1,spiketimes_mnp1)
        popfunc.spike_report("IPSILATERAL MNP EXT",  senders_mnp2, spiketimes_mnp2)

        popfunc.spike_report("CONTRALATERAL MNP FLX", senders_contra_mnp1, spiketimes_contra_mnp1)
        popfunc.spike_report("CONTRALATERAL MNP FLX", senders_contra_mnp2, spiketimes_contra_mnp2)
        

    if nn.args['low_locomotion_v0v_right'] and nn.args['low_locomotion_v0v_right']:

        popfunc.spike_report("RG FLX IPSILATERAL", senders_rc_1,       spiketimes_rc_1)
        popfunc.spike_report("V2A IPSILATERAL", senders_exc_inter_tonic1, spiketimes_exc_inter_tonic1)
        popfunc.spike_report("V0v IPSILATERAL",    senders_V0v,        spiketimes_V0v)
        popfunc.spike_report("RG FLX CONTRALATERAL EXC BURST", senders_contra_rg_exc_burst1, spiketimes_contra_rg_exc_burst1)

        popfunc.spike_report("V2A CONTRALATERAL", senders_contra_exc_inter_tonic1, spiketimes_contra_exc_inter_tonic1)
        popfunc.spike_report("V0v CONTRALATERAL",  senders_V0v_contra, spiketimes_V0v_contra)

        popfunc.spike_report("IPSILATERAL MNP FLX", senders_mnp1,spiketimes_mnp1)
        popfunc.spike_report("IPSILATERAL MNP EXT",  senders_mnp2, spiketimes_mnp2)

        popfunc.spike_report("CONTRALATERAL MNP FLX", senders_contra_mnp1, spiketimes_contra_mnp1)
        popfunc.spike_report("CONTRALATERAL MNP FLX", senders_contra_mnp2, spiketimes_contra_mnp2)
        

    if nn.fb_rg_flx == 1:
        #Read spike data - poisson generators
        senders_rg_flx_pg,spiketimes_rg_flx_pg = popfunc.read_spike_data(rg1.spike_detector_rg_flx_pg)

    # inh1 = V2b 
    # inh2 = v1 
    # inh2_contra = v1_contra 
    #Read spike data - V1/V2b inhibitory populations
    if nn.rgs_connected==1:
    
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
            spike_bins_inh_inter_tonic1 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1)
            spike_bins_inh_inter1 = spike_bins_inh_inter_tonic1
            
            spike_bins_inh_inter1_true = spike_bins_inh_inter1
            spike_bins_inh_inter1 = (spike_bins_inh_inter1-np.min(spike_bins_inh_inter1))/(np.max(spike_bins_inh_inter1)-np.min(spike_bins_inh_inter1))
            
            spike_bins_inh_inter_tonic2 = popfunc.rate_code_spikes(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)
            spike_bins_inh_inter2 = spike_bins_inh_inter_tonic2
            
            spike_bins_inh_inter2_true = spike_bins_inh_inter2
            spike_bins_inh_inter2 = (spike_bins_inh_inter2-np.min(spike_bins_inh_inter2))/(np.max(spike_bins_inh_inter2)-np.min(spike_bins_inh_inter2))
            
        t_stop = time.perf_counter()
        print('Rate coded activity complete, taking ',int(t_stop-t_start),' seconds.')

        print('Moving onto plotting functions...')

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
        ax[0].plot(t, spike_bins_rg2_true)
        ax[1].plot(t, spike_bins_mnp1_true)
        ax[1].plot(t, spike_bins_mnp2_true)
        ax[0].set_xticks([])
        ax[0].set_xlim(0,len(spike_bins_rg1_true))
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000])
        ax[1].set_xticklabels([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        ax[1].set_xlim(0,len(spike_bins_rg1_true))
        ax[0].legend(['RG_F', 'RG_E'],loc='upper right',fontsize='x-small')  
        ax[1].legend(['FLX', 'EXT'],loc='upper right',fontsize='x-small')
        ax[0].set_title(f"Population output (RG) - {label}")
        ax[1].set_title(f"Population output (MNP) - {label}")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + f'{label}_rate_coded_output_rg_mnp.png',bbox_inches="tight")
            
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
        print(f"RG1 Exc (bursting): {np.nanmean([np.nanmean(f) for f in rgexc1_bursting_freq]):.2f} Hz")
        print(f"RG2 Exc (bursting): {np.nanmean([np.nanmean(f) for f in rgexc2_bursting_freq]):.2f} Hz")

        print(f"RG1 Exc (tonic): {np.nanmean([np.nanmean(f) for f in rgexc1_tonic_freq]):.2f} Hz")
        print(f"RG2 Exc (tonic): {np.nanmean([np.nanmean(f) for f in rgexc2_tonic_freq]):.2f} Hz")

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

        v0v_freq, v0v_times = popfunc.calculate_interspike_frequency(nn.v0v_pop_size,spiketimes_V0v)
        v0d_freq, v0d_times = popfunc.calculate_interspike_frequency(nn.v0d_pop_size,spiketimes_V0d)
        
        print(f"V0c (Flexor-Connected) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0c1_freq]):.2f} Hz")
        print(f"V0c (Extensor-Connected) (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0c2_freq]):.2f} Hz")

        print(f"V0v (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0v_freq]):.2f} Hz")
        print(f"V0d (Ipsilateral): {np.nanmean([np.nanmean(f) for f in v0d_freq]):.2f} Hz")

        v0d_contra_freq, v0d_contra_times = popfunc.calculate_interspike_frequency(nn.v0d_pop_size, spiketimes_V0d_contra)
        v0v_contra_freq, v0v_contra_times = popfunc.calculate_interspike_frequency(nn.v0v_pop_size, spiketimes_V0v_contra)

        print(f"V0v (Contralateral): {np.nanmean([np.nanmean(f) for f in v0v_contra_freq]):.2f} Hz")
        print(f"V0d (Contralateral): {np.nanmean([np.nanmean(f) for f in v0d_contra_freq]):.2f} Hz")

        v1a1_freq, v1a1_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_1)
        v1a2_freq, v1a2_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_2)
        
        rc1_freq, rc1_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_1)
        rc2_freq, rc2_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_2)
        
        if nn.rgs_connected:
            v2b_freq, v2b_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1)

            v1_freq, v1_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)

            print(f"V1 (ipsilateral): {np.nanmean([np.nanmean(f) for f in v1_freq]):.2f} Hz")

            v1_contra_freq, v1_contra_times = popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1,spikestimes_inh_inter_tonic2_contra)
     
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

        #Convolve spike data - v2a and v0v 

        if nn.low_locomotion_v0d_left and nn.low_locomotion_v0d_right: 
        
            v0d_contra_convolved, _, v0d_contra_neuron_convolved = popfunc.convolve_spiking_activity(nn.v0d_pop_size, spiketimes_V0d_contra)
            v0d_convolved, _, v0d_neuron_convolved = popfunc.convolve_spiking_activity(nn.v0d_pop_size,spiketimes_V0d)

       
        if nn.low_locomotion_v0v_left and nn.low_locomotion_v0v_right: 

            v0v_convolved, _, v0v_neuron_convolved = popfunc.convolve_spiking_activity(nn.v0v_pop_size,spiketimes_V0v)   # population-averaged signal (T, )
            v0v_contra_convolved, _, v0v_contra_neuron_convolved = popfunc.convolve_spiking_activity(nn.v0v_pop_size, spiketimes_V0v_contra)

        #Convolve spike data - MNPs
        mnp1_convolved, convolved_time, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp1)
        mnp2_convolved, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp2)

        mnp1_convolved_contra, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_contra_mnp1)
        mnp2_convolved_contra, _, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_contra_mnp2)

        # --- Convolve CONTRALATERAL RG populations (full symmetry to ipsilateral) ---

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

        if nn.low_locomotion_v0v_left and nn.low_locomotion_v0v_left: 
       
            v0v_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0v_freq]))
            V0v_conv_max = np.nanmax(v0v_convolved)
            v0v_contra_isf_max = np.nanmax([np.nanmean(f) for f in v0v_contra_freq])
            
            print('Max Firing rate of V0V (Ipsilateral) (ISF):', round(v0v_isf_max,2), ' Max Firing rate of V0D (Contralateral) (ISF)', round(v0v_contra_isf_max,2))

            v0v_contra_scale = v0v_contra_isf_max / np.nanmax(v0v_contra_convolved)
            v0v_scale = v0v_isf_max / V0v_conv_max
          
                
            
        if nn.low_locomotion_v0d_left and nn.low_locomotion_v0d_right: 
       
            v0d_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0d_freq]))
            v0d_contra_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0d_contra_freq]))
            V0d_conv_max = np.nanmax(v0d_convolved)
            V0d_contra_conv_max = np.nanmax(v0d_contra_convolved)

            print('Max Firing rate of V0D (Ipsilateral) (ISF):', round(v0d_isf_max,2), ' Max Firing rate of V0D (Contralateral) (ISF)', round(v0d_contra_isf_max,2))
            
            v0d_scale = v0d_isf_max / V0d_conv_max
            v0d_contra_scale = v0d_contra_isf_max / V0d_contra_conv_max
            

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
        
        print('[INFO] After scaling max firing rate of a Flx MN (Convolved):',round(mnp1_convolved_max_scaled,2),'Ext MN:',round(mnp2_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx MN (Convolved):',round(mnp1_convolved_scaled_mean,2),'Ext MN:',round(mnp2_convolved_scaled_mean,2))
        
          # prepping variables for analyze_output function 
        mnp1_avg_norm = (mnp1_convolved-np.min(mnp1_convolved))/(np.max(mnp1_convolved)-np.min(mnp1_convolved))
        mnp2_avg_norm = (mnp2_convolved-np.min(mnp2_convolved))/(np.max(mnp2_convolved)-np.min(mnp2_convolved))

        
        # Scaling mnp1 & mnp2 contra 
        mnp1_contra_convolved_scaled = mnp1_convolved_contra * mnp1_scale
        mnp2_conta_convolved_scaled = mnp2_convolved_contra * mnp2_scale
        
        mnp1_contra_convolved_scaled_mean = np.nanmean(mnp1_contra_convolved_scaled)
        mnp2_conta_convolved_scaled_mean = np.nanmean(mnp2_conta_convolved_scaled)
        mnp1_contra_convolved_max_scaled = np.nanmax(mnp1_convolved_contra * mnp1_scale)
        mnp2_conta_convolved_max_scaled = np.nanmax(mnp2_convolved_contra * mnp1_scale)

        print('[INFO] After scaling max firing rate of a Flx CONTRA  MN (Convolved):',round(mnp1_contra_convolved_max_scaled,2),'Ext MN:',round(mnp2_conta_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx CONTRA MN (Convolved):',round(mnp1_contra_convolved_scaled_mean,2),'Ext MN:',round(mnp2_conta_convolved_scaled_mean,2))
        
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
    
        print('[INFO] After scaling max firing rate of a Flx RG (Convolved):',round(rg_1_contra_convolved_max_scaled,2),'Ext RG:',round(rg_2_contra_convolved_max_scaled,2))
        print('[INFO] After scaling mean firing rate of a Flx MN (Convolved):',round(rg1_contra_convolved_scaled_mean,2),'Ext MN:',round(rg2_contra_convolved_scaled_mean,2))
        
        rg1_contra_avg_norm = (contra_rg1_convolved-np.min(contra_rg1_convolved))/(np.max(contra_rg1_convolved)-np.min(contra_rg1_convolved))
        rg2_contra_avg_norm = (contra_rg2_convolved-np.min(contra_rg2_convolved))/(np.max(contra_rg2_convolved)-np.min(contra_rg2_convolved))

        # Scaling rg1 and rg2 contra 
        if max(mnp1_avg_norm)>0 and max(mnp2_avg_norm)>0: 

            print("[INFO] Calling Analyze Output Function for MNP (Ipsilateral) ")
            
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(mnp1_avg_norm,mnp2_avg_norm,mnp1_convolved_scaled,mnp2_convolved_scaled,'MNP',y_line_bd=0.4,y_line_phase=0.7)

       
        if max(mnp1_contra_avg_norm) > 0 and max(mnp2_contra_avg_norm) > 0: 

            print("[INFO] Calling Analyze Output Function for MNP (Contralateral)")

            avg_freq, avg_phase, bd_comparison = calc.analyze_output(mnp1_avg_norm,mnp2_avg_norm,mnp1_convolved_scaled,mnp2_convolved_scaled,'MNP',y_line_bd=0.4,y_line_phase=0.7)

    
       
        if max(rg1_avg_norm) > 0 and max(rg2_avg_norm) > 0:  

            print("[INFO] Calling Analyze Output Function for RG (Ipsilateral)")

            avg_freq, avg_phase, bd_comparison = calc.analyze_output(rg1_avg_norm,rg2_avg_norm,rg1_convolved_scaled,rg2_convolved_scaled,'RG',y_line_bd=0.4,y_line_phase=0.7)

        if max(rg1_contra_avg_norm) > 0 and max(rg2_contra_avg_norm) > 0: 

            print("[INFO] Calling Analyze Output Function for RG (Contralateral)")
        
     

    
        
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


        # v0d plot 
        if nn.args['low_locomotion_v0d_right'] and nn.args['low_locomotion_v0d_left']:

            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.2])  # RG, V0d, MNP

            # --- Row 1: RG ---
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

    
            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle='--', alpha=0.6)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle='--', alpha=0.6)
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")


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

        # v0v plot 
         # v0v plot 
        if nn.args['low_locomotion_v0v_right'] and nn.args['low_locomotion_v0v_left']:

            from matplotlib import gridspec

            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(5, 2, height_ratios=[1.2, 1.0, 1.0, 1.2, 1.0])  
            # Rows: RG, V2a, V1, V0v, MNP

            # ----------------------------------------------------------
            # ROW 1: RG
            # ----------------------------------------------------------
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.set_title("RG Left")
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle='--', alpha=0.7)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle='--', alpha=0.7)
            ax_rg_right.set_title("RG Right")
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")

            # ----------------------------------------------------------
            # ROW 2: V2a (NEW ROW)
            # ----------------------------------------------------------
            ax_v2a_left  = fig.add_subplot(gs[1, 0])
            ax_v2a_right = fig.add_subplot(gs[1, 1])

            ax_v2a_left.plot(t, v2a1_convolved * v2a1_scale, color="tab:green")
            ax_v2a_left.set_title("V2a Left (ipsilateral)")
            ax_v2a_left.set_ylabel("Freq (Hz)")

            ax_v2a_right.plot(t, v2a_contra_convolved * contra_v2a_scale, color="tab:green")
            ax_v2a_right.set_title("V2a Right (contralateral)")

    
            # ----------------------------------------------------------
            # ROW 3: V0v
            # ----------------------------------------------------------
            ax_v0v_left = fig.add_subplot(gs[2, 0])
            ax_v0v_right = fig.add_subplot(gs[2, 1])

            ax_v0v_left.plot(t, v0v_convolved * v0v_scale, color="tab:blue")
            ax_v0v_left.set_title("V0v Left (ipsilateral)")

            ax_v0v_right.plot(t, v0v_contra_convolved * v0v_contra_scale, color="tab:red")
            ax_v0v_right.set_title("V0v Right (contralateral)")

            # ----------------------------------------------------------
            # ROW 4: V1
            # ----------------------------------------------------------
            
            ax_v1_left = fig.add_subplot(gs[3, 0])
            ax_v1_right = fig.add_subplot(gs[3, 1])

            ax_v1_left.plot(t, v1_convolved * v1_scale, color="tab:blue")
            ax_v1_left.set_title("V1 Left Hemicord (Ipsilateral)")

            ax_v1_right.plot(t, v1_contra_convolved * v1_scale_contra, color="tab:red")
            ax_v1_right.set_title("V1 Right Hemicord (Contralateral)")


            # ----------------------------------------------------------
            # ROW 4: MNP
            # ----------------------------------------------------------
            ax_mnp_left = fig.add_subplot(gs[4, 0])
            ax_mnp_right = fig.add_subplot(gs[4, 1])

            ax_mnp_left.plot(t, mnp1_convolved * mnp1_scale)
            ax_mnp_left.plot(t, mnp2_convolved * mnp2_scale)
            ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small")
            ax_mnp_left.set_title("MNP Left")
            ax_mnp_left.set_xlabel("Time (ms)")

            ax_mnp_right.plot(t, contra_mnp1_convolved * contra_mnp1_scale, linestyle='--', alpha=0.7)
            ax_mnp_right.plot(t, contra_mnp2_convolved * contra_mnp2_scale, linestyle='--', alpha=0.7)
            ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small")
            ax_mnp_right.set_title("MNP Right")
            ax_mnp_right.set_xlabel("Time (ms)")

            plt.tight_layout()

            if nn.args['save_results']:
                plt.savefig(nn.pathFigures + '/' + f"{label}_RG_V2a_V0v_MNP_combined.png",
                            dpi=300, bbox_inches="tight")

            plt.close()

        

        if nn.args['heatmap_recruitment_plot']: 

            def plot_recruitment_heatmap(
                title,
                convolved_activity,   # (neurons × time)
                time_vec,
                save_path=None
            ):
                # Sort neurons by mean firing rate
                sorted_activity = convolved_activity


                pop_mean = np.nanmean(convolved_activity, axis=0)

                fig, ax = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

                ax[0].plot(time_vec, pop_mean)
                ax[0].set_ylabel("Population firing rate (a.u.)")
                ax[0].set_title(title)

                im = ax[1].imshow(
                    sorted_activity,
                    aspect="auto",
                    origin="lower",
                    extent=[time_vec[0], time_vec[-1], 0, sorted_activity.shape[0]],
                    cmap="plasma"
                )

                ax[1].set_ylabel("Neuron ID")
                ax[1].set_xlabel("Time (ms)")
                fig.colorbar(im, ax=ax[1], label="Convolved firing rate")

                plt.tight_layout()

                if save_path is not None:
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")

                plt.show()
        
            if nn.args['low_locomotion_v0d_left'] & nn.args['low_locomotion_v0d_right']:
                plot_recruitment_heatmap(
                    "V0d Ipsilateral Recruitment",
                    v0d_neuron_convolved,
                    t,
                    save_path=f"{nn.pathFigures}/{label}_V0d_ipsi_recruitment.png"
                )

                plot_recruitment_heatmap(
                    "V0d Contralateral Recruitment",
                    v0d_contra_neuron_convolved,
                    t,
                    save_path=f"{nn.pathFigures}/{label}_V0d_contra_recruitment.png"
                )


            if nn.args['low_locomotion_v0v_left'] & nn.args['low_locomotion_v0v_right']:
                plot_recruitment_heatmap(
                    "V0v Ipsilateral Recruitment",
                    v0v_neuron_convolved,
                    t,
                    save_path=f"{nn.pathFigures}/{label}_V0v_ipsi_recruitment.png"
                )

                plot_recruitment_heatmap(
                    "V0v Contralateral Recruitment",
                    v0v_contra_neuron_convolved,
                    t,
                    save_path=f"{nn.pathFigures}/{label}_V0v_contra_recruitment.png"
                )


            if nn.args['online_ramp_heatmap_recruitment_plot']: 

                import numpy as np
                import matplotlib.pyplot as plt
                import matplotlib.ticker as mticker

                def plot_recruitment_heatmap_boxes(
                    *,
                    title: str,
                    convolved_activity: np.ndarray,   # shape: (n_neurons, n_time)
                    time_vec: np.ndarray,             # shape: (n_time,)
                    neuron_ids=None,                  # list/array length n_neurons (optional)
                    ramp_weight_log=None,             # dict {"time":[...], "weight":[...]} optional
                    cmap: str = "plasma",
                    save_path: str | None = None,
                    show: bool = True,
                    max_y_ticks: int = 12
                ):
                    """
                    Heatmap where each pixel = one neuron's activity at one time bin.
                    No population mean plot.
                    Optionally adds weight-vs-time panel underneath for online ramp.
                    """

                    A = np.asarray(convolved_activity, dtype=float)
                    t = np.asarray(time_vec, dtype=float)

                    if A.ndim != 2:
                        raise ValueError(f"convolved_activity must be 2D (neurons x time). Got shape {A.shape}")
                    if t.ndim != 1:
                        raise ValueError(f"time_vec must be 1D. Got shape {t.shape}")
                    if A.shape[1] != t.size:
                        raise ValueError(f"Mismatch: activity has {A.shape[1]} timepoints but time_vec has {t.size}")

                    n_neurons, n_time = A.shape

                    # Labels
                    if neuron_ids is None:
                        neuron_ids = np.arange(n_neurons)
                        y_label = "Neuron index"
                    else:
                        neuron_ids = np.asarray(neuron_ids)
                        if neuron_ids.size != n_neurons:
                            raise ValueError(f"neuron_ids length {neuron_ids.size} != n_neurons {n_neurons}")
                        y_label = "Neuron ID"

                    # Optional weight interpolation onto time_vec
                    ramp_w_interp = None
                    if ramp_weight_log is not None:
                        ramp_t = np.asarray(ramp_weight_log.get("time", []), dtype=float)
                        ramp_w = np.asarray(ramp_weight_log.get("weight", []), dtype=float)
                        if ramp_t.size > 1 and ramp_w.size == ramp_t.size:
                            order = np.argsort(ramp_t)
                            ramp_t = ramp_t[order]
                            ramp_w = ramp_w[order]
                            ramp_w_interp = np.interp(t, ramp_t, ramp_w, left=ramp_w[0], right=ramp_w[-1])

                    # Layout
                    nrows = 2 if ramp_w_interp is not None else 1
                    height_ratios = [3.0, 1.0] if nrows == 2 else [1.0]

                    fig, axes = plt.subplots(
                        nrows, 1,
                        figsize=(18, 7 if nrows == 1 else 10),
                        sharex=True,
                        gridspec_kw={"height_ratios": height_ratios}
                    )
                    if nrows == 1:
                        ax_hm = axes
                        ax_w = None
                    else:
                        ax_hm, ax_w = axes

                    # Heatmap (pixel boxes)
                    im = ax_hm.imshow(
                        A,
                        aspect="auto",
                        origin="lower",
                        extent=[t[0], t[-1], 0, n_neurons],
                        cmap=cmap,
                        interpolation="nearest"   # IMPORTANT: keeps pixel boxes sharp
                    )
                    ax_hm.set_title(title)
                    ax_hm.set_ylabel(y_label)

                    # y ticks (don’t label thousands of neurons)
                    if n_neurons <= max_y_ticks:
                        idx = np.arange(n_neurons)
                    else:
                        idx = np.linspace(0, n_neurons - 1, max_y_ticks).astype(int)

                    ax_hm.set_yticks(idx + 0.5)
                    ax_hm.set_yticklabels([str(neuron_ids[i]) for i in idx])

                    cbar = fig.colorbar(im, ax=ax_hm, pad=0.01)
                    cbar.set_label("Convolved firing rate")

                    # Optional weight plot
                    if ax_w is not None:
                        ax_w.plot(t, ramp_w_interp)
                        ax_w.set_title("Synaptic weight over time")
                        ax_w.set_ylabel("Weight")
                        ax_w.set_xlabel("Time (ms)")
                        ax_w.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

                    else:
                        ax_hm.set_xlabel("Time (ms)")

                    fig.tight_layout()

                    if save_path is not None:
                        plt.savefig(save_path, dpi=300, bbox_inches="tight")

                    if show:
                        plt.show()
                    else:
                        plt.close(fig)

                    return fig
                
                
                if nn.args['low_locomotion_v0d_left'] & nn.args['low_locomotion_v0d_right']:

                    pop_mean, time_vec, smoothed_spikes = popfunc.convolve_spiking_activity(pop_size, spiketimes)

                    plot_recruitment_heatmap_boxes(
                        title=f"{label} | V0d recruitment (ONLINE RAMP) | {ramp_weight_name}",
                        convolved_activity=smoothed_spikes,
                        time_vec=time_vec,
                        neuron_ids=np.array(L_V0D.v0d_bursting),   # <- real IDs if available
                        ramp_weight_log=online_ramp_log,           # optional weight trace
                        save_path=nn.pathFigures + f"/{label}_V0d_recruitment_boxes.png",
                    )

                if nn.args['low_locomotion_v0v_left'] & nn.args['low_locomotion_v0v_right']:

                    pop_mean, time_vec, smoothed_spikes = popfunc.convolve_spiking_activity(pop_size, spiketimes)

                    plot_recruitment_heatmap_boxes(
                        title=f"{label} | V0d recruitment (ONLINE RAMP) | {ramp_weight_name}",
                        convolved_activity=smoothed_spikes,
                        time_vec=time_vec,
                        neuron_ids=np.array(L_V0D.v0d_bursting),   # <- real IDs if available
                        ramp_weight_log=online_ramp_log,           # optional weight trace
                        save_path=nn.pathFigures + f"/{label}_V0d_recruitment_boxes.png",
                    )




            # if nn.args['offline_ramp_heatmap_recruitment_plot']:


            #     def plot_recruitment_heatmap_offline_ramp(
            #         title,
            #         convolved_activity,   # (neurons × time)
            #         time_vec,
            #         save_path=None
            #     ):
            #         # Sort neurons by mean firing rate
            #         sorted_activity = convolved_activity

            #         # should be Neuron ID1, Neuron ID2 ... we should just take them.
            #         # and this should be the label on the y-axis. 
            #         # x-axis should be the time of the experiment. 
            #         # since this is an offline ramp we are also going to have a label on the weight? 
            #   ax_w = fig.add_subplot(gs[5, :])
            # ax_w.plot(t, ramp_w_interp)

            # ax_w.set_title(f"{ramp_weight_name} (weight over time)")
            # ax_w.set_xlabel("Time (ms)")
            # ax_w.set_ylabel("Weight")

            # # ---- FORCE more Y-axis ticks ----
            # w_min = np.min(ramp_w_interp)
            # w_max = np.max(ramp_w_interp)

            # ax_w.set_ylim(w_min, w_max)

            # # Put a tick every 0.25 (or 0.1 / 0.5 depending on scale)
            # tick_step = 0.5
            # ax_w.set_yticks(np.arange(
            #     np.floor(w_min / tick_step) * tick_step,
            #     np.ceil(w_max / tick_step) * tick_step + tick_step,
            #     tick_step
            # ))

            # ax_w.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            # # Save once
            # if nn.args["save_results"]:
            #     plt.savefig(
            #         nn.pathFigures + f"/{label}_ONLINE_RAMP_RG_V2a_V0v_V1_MNP_WEIGHT.png",
            #         dpi=300,
            #         bbox_inches="tight"
            #     )

            # plt.close(fig)


            #         pop_mean = np.nanmean(convolved_activity, axis=0)

            #         fig, ax = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

            #         ax[0].plot(time_vec, pop_mean)
            #         ax[0].set_ylabel("Population firing rate (a.u.)")
            #         ax[0].set_title(title)

            #         im = ax[1].imshow(
            #             sorted_activity,
            #             aspect="auto",
            #             origin="lower",
            #             extent=[time_vec[0], time_vec[-1], 0, sorted_activity.shape[0]],
            #             cmap="plasma"
            #         )

            #         ax[1].set_ylabel("Neuron ID")
            #         ax[1].set_xlabel("Time (ms)")
            #         fig.colorbar(im, ax=ax[1], label="Convolved firing rate")

            #         plt.tight_layout()

            #         if save_path is not None:
            #             plt.savefig(save_path, dpi=300, bbox_inches="tight")

            #         plt.show()




        if ramp_type == "offline" and ramp_weight is not None and nn.args['low_locomotion_v0v_left'] & nn.args['low_locomotion_v0v_right']:

            print("===============================================")
            print("[INFO] V0V Offline Ramp Plotting In Progress...")
            print("===============================================")
            time.sleep(5)

            from matplotlib import gridspec

            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(5, 2, height_ratios=[1.2, 1.0, 1.0, 1.2, 1.0])  
            # Rows: RG, V2a, V1, V0v, MNP

            fig.suptitle(
                f"{label} | OFFLINE RAMP | {ramp_weight_name} = {ramp_weight:.2f}",
                fontsize=14,
                y=0.98
            )

            # ----------------------------------------------------------
            # ROW 1: RG
            # ----------------------------------------------------------
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.set_title("RG Left")
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle='--', alpha=0.7)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle='--', alpha=0.7)
            ax_rg_right.set_title("RG Right")
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")

            # ----------------------------------------------------------
            # ROW 2: V2a (NEW ROW)
            # ----------------------------------------------------------
            ax_v2a_left  = fig.add_subplot(gs[1, 0])
            ax_v2a_right = fig.add_subplot(gs[1, 1])

            ax_v2a_left.plot(t, v2a1_convolved * v2a1_scale, color="tab:green")
            ax_v2a_left.set_title("V2a Left (ipsilateral)")
            ax_v2a_left.set_ylabel("Freq (Hz)")

            ax_v2a_right.plot(t, v2a_contra_convolved * contra_v2a_scale, color="tab:green")
            ax_v2a_right.set_title("V2a Right (contralateral)")

    
            # ----------------------------------------------------------
            # ROW 3: V0v
            # ----------------------------------------------------------
            ax_v0v_left = fig.add_subplot(gs[2, 0])
            ax_v0v_right = fig.add_subplot(gs[2, 1])

            ax_v0v_left.plot(t, v0v_convolved * v0v_scale, color="tab:blue")
            ax_v0v_left.set_title("V0v Left (ipsilateral)")

            ax_v0v_right.plot(t, v0v_contra_convolved * v0v_contra_scale, color="tab:red")
            ax_v0v_right.set_title("V0v Right (contralateral)")

            # ----------------------------------------------------------
            # ROW 4: V1
            # ----------------------------------------------------------
            
            ax_v1_left = fig.add_subplot(gs[3, 0])
            ax_v1_right = fig.add_subplot(gs[3, 1])

            ax_v1_left.plot(t, v1_convolved * v1_scale, color="tab:blue")
            ax_v1_left.set_title("V1 Left Hemicord (Ipsilateral)")

            ax_v1_right.plot(t, v1_contra_convolved * v1_scale_contra, color="tab:red")
            ax_v1_right.set_title("V1 Right Hemicord (Contralateral)")


            # ----------------------------------------------------------
            # ROW 4: MNP
            # ----------------------------------------------------------
            ax_mnp_left = fig.add_subplot(gs[4, 0])
            ax_mnp_right = fig.add_subplot(gs[4, 1])

            ax_mnp_left.plot(t, mnp1_convolved * mnp1_scale)
            ax_mnp_left.plot(t, mnp2_convolved * mnp2_scale)
            ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small")
            ax_mnp_left.set_title("MNP Left")
            ax_mnp_left.set_xlabel("Time (ms)")

            ax_mnp_right.plot(t, contra_mnp1_convolved * contra_mnp1_scale, linestyle='--', alpha=0.7)
            ax_mnp_right.plot(t, contra_mnp2_convolved * contra_mnp2_scale, linestyle='--', alpha=0.7)
            ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small")
            ax_mnp_right.set_title("MNP Right")
            ax_mnp_right.set_xlabel("Time (ms)")

            plt.tight_layout()

            if nn.args['save_results']:
                plt.savefig(nn.pathFigures + '/' + f"{label}_RG_V2a_V0v_MNP_combined.png",
                            dpi=300, bbox_inches="tight")

            plt.close()


        if ramp_type == "offline" and ramp_weight is not None and nn.args['low_locomotion_v0d_left'] and nn.args['low_locomotion_v0d_right']:

            print("[INFO] V0D Offline Ramp Plotting In Progress.")
        
            print("[PLOT DEBUG] ramp_w min/max:", np.min(ramp_w), np.max(ramp_w))
            print("[PLOT DEBUG] first/last:", ramp_w[0], ramp_w[-1])
            time.sleep(3)
            

            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.2])  # RG, V0d, MNP

            fig.suptitle(
                f"{label} | OFFLINE RAMP | {ramp_weight_name} = {ramp_weight:.2f}",
                fontsize=14,
                y=0.98
            )
         
            # --- Row 1: RG ---
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

    
            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle='--', alpha=0.6)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle='--', alpha=0.6)
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")


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



    if (
        ramp_type == "online"
        and ramp_weight is not None
        and nn.args["low_locomotion_v0v_left"]
        and nn.args["low_locomotion_v0v_right"]
    ):
        
        from matplotlib import gridspec
        import matplotlib.ticker as mticker

        print("[INFO] V0V Online Ramp Plotting In Progress.")
        time.sleep(5)

        ramp_t = np.asarray(ramp_weight["time"], dtype=float)
        ramp_w = np.asarray(ramp_weight["weight"], dtype=float)

        # Safety
        if ramp_t.size == 0 or ramp_w.size == 0:
            print("[WARN] ramp_weight log is empty; skipping ONLINE ramp plot.")
        elif ramp_t.size != ramp_w.size:
            print(f"[WARN] ramp_t and ramp_w length mismatch: {ramp_t.size} vs {ramp_w.size}; skipping plot.")
        else:
            # Sort for interpolation
            order = np.argsort(ramp_t)
            ramp_t = ramp_t[order]
            ramp_w = ramp_w[order]

            t = np.asarray(t, dtype=float)

            # Interpolate weight onto the same time base as your convolved traces
            ramp_w_interp = np.interp(t, ramp_t, ramp_w)

            # -------- Figure layout: add a bottom row for weight --------
            fig = plt.figure(figsize=(16, 15), constrained_layout=True)
            gs = gridspec.GridSpec(
                6, 2,
                height_ratios=[1.2, 1.0, 1.0, 1.0, 1.2, 0.9],  # RG, V2a, V0v, V1, MNP, Weight
                figure=fig
            )

            fig.suptitle(
                f"{label} | ONLINE RAMP | {ramp_weight_name} "
                f"({np.nanmin(ramp_w):.2f} → {np.nanmax(ramp_w):.2f})",
                fontsize=20
            )

            # ----------------------------------------------------------
            # ROW 1: RG
            # ----------------------------------------------------------
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.set_title("RG Left (ipsilateral)")
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle="--", alpha=0.7)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle="--", alpha=0.7)
            ax_rg_right.set_title("RG Right (contralateral)")
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")

            # ----------------------------------------------------------
            # ROW 2: V2a
            # ----------------------------------------------------------
            ax_v2a_left  = fig.add_subplot(gs[1, 0])
            ax_v2a_right = fig.add_subplot(gs[1, 1])

            ax_v2a_left.plot(t, v2a1_convolved * v2a1_scale)
            ax_v2a_left.set_title("V2a Left (ipsilateral)")
            ax_v2a_left.set_ylabel("Freq (Hz)")

            ax_v2a_right.plot(t, v2a_contra_convolved * contra_v2a_scale)
            ax_v2a_right.set_title("V2a Right (contralateral)")

            # ----------------------------------------------------------
            # ROW 3: V0v (plain lines; no color)
            # ----------------------------------------------------------
            ax_v0v_left  = fig.add_subplot(gs[2, 0])
            ax_v0v_right = fig.add_subplot(gs[2, 1])

            ax_v0v_left.plot(t, v0v_convolved * v0v_scale)
            ax_v0v_left.set_title("V0v Left (ipsilateral)")
            ax_v0v_left.set_ylabel("Freq (Hz)")

            ax_v0v_right.plot(t, v0v_contra_convolved * v0v_contra_scale)
            ax_v0v_right.set_title("V0v Right (contralateral)")

            # Keep V0v y-lims comparable between ipsi/contra
            v0v_left_y  = v0v_convolved * v0v_scale
            v0v_right_y = v0v_contra_convolved * v0v_contra_scale
            ymax = np.nanmax([np.nanmax(v0v_left_y), np.nanmax(v0v_right_y)])
            if np.isfinite(ymax) and ymax > 0:
                ax_v0v_left.set_ylim(0, ymax * 1.1)
                ax_v0v_right.set_ylim(0, ymax * 1.1)

            # ----------------------------------------------------------
            # ROW 4: V1
            # ----------------------------------------------------------
            ax_v1_left  = fig.add_subplot(gs[3, 0])
            ax_v1_right = fig.add_subplot(gs[3, 1])

            ax_v1_left.plot(t, v1_convolved * v1_scale)
            ax_v1_left.set_title("V1 Left Hemicord (Ipsilateral)")
            ax_v1_left.set_ylabel("Freq (Hz)")

            ax_v1_right.plot(t, v1_contra_convolved * v1_scale_contra)
            ax_v1_right.set_title("V1 Right Hemicord (Contralateral)")

            # ----------------------------------------------------------
            # ROW 5: MNP
            # ----------------------------------------------------------
            ax_mnp_left  = fig.add_subplot(gs[4, 0])
            ax_mnp_right = fig.add_subplot(gs[4, 1])

            ax_mnp_left.plot(t, mnp1_convolved * mnp1_scale)
            ax_mnp_left.plot(t, mnp2_convolved * mnp2_scale)
            ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small")
            ax_mnp_left.set_title("MNP Left")
            ax_mnp_left.set_xlabel("Time (ms)")

            ax_mnp_right.plot(t, contra_mnp1_convolved * contra_mnp1_scale, linestyle="--", alpha=0.7)
            ax_mnp_right.plot(t, contra_mnp2_convolved * contra_mnp2_scale, linestyle="--", alpha=0.7)
            ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small")
            ax_mnp_right.set_title("MNP Right")
            ax_mnp_right.set_xlabel("Time (ms)")

            # ----------------------------------------------------------
            # ROW 6: Weight vs time (spans both columns)
            # ----------------------------------------------------------
            ax_w = fig.add_subplot(gs[5, :])
            ax_w.plot(t, ramp_w_interp)

            ax_w.set_title(f"{ramp_weight_name} (weight over time)")
            ax_w.set_xlabel("Time (ms)")
            ax_w.set_ylabel("Weight")

            # ---- FORCE more Y-axis ticks ----
            w_min = np.min(ramp_w_interp)
            w_max = np.max(ramp_w_interp)

            ax_w.set_ylim(w_min, w_max)

            # Put a tick every 0.25 (or 0.1 / 0.5 depending on scale)
            tick_step = 0.5
            ax_w.set_yticks(np.arange(
                np.floor(w_min / tick_step) * tick_step,
                np.ceil(w_max / tick_step) * tick_step + tick_step,
                tick_step
            ))

            ax_w.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            # Save once
            if nn.args["save_results"]:
                plt.savefig(
                    nn.pathFigures + f"/{label}_ONLINE_RAMP_RG_V2a_V0v_V1_MNP_WEIGHT.png",
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.close(fig)

    else:
        ramp_t = None
        ramp_w = None




    if (
        ramp_type == "online"
        and ramp_weight is not None
        and nn.args["low_locomotion_v0d_left"]
        and nn.args["low_locomotion_v0d_right"]
    ):
        print("[INFO] V0D Online Ramp Plotting In Progress...")

        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mticker

        ramp_t = np.asarray(ramp_weight["time"], dtype=float)
        ramp_w = np.asarray(ramp_weight["weight"], dtype=float)

        # Safety
        if ramp_t.size == 0 or ramp_w.size == 0:
            print("[WARN] ramp_weight log is empty; skipping ONLINE ramp plot.")
        elif ramp_t.size != ramp_w.size:
            print(f"[WARN] ramp_t and ramp_w length mismatch: {ramp_t.size} vs {ramp_w.size}; skipping plot.")
        else:
            # Ensure sorted times for interpolation
            order = np.argsort(ramp_t)
            ramp_t = ramp_t[order]
            ramp_w = ramp_w[order]

            t = np.asarray(t, dtype=float)

            # print("[DEBUG] t head:", t[:5], "tail:", t[-5:])
            # print("[DEBUG] ramp_t head:", ramp_t[:5], "tail:", ramp_t[-5:])
            # print("[DEBUG] ramp_t range:", ramp_t.min(), "→", ramp_t.max())
            # print("[DEBUG] ramp_w range:", ramp_w.min(), "→", ramp_w.max())
            # time.sleep(10)

            # Interpolate ramp weights onto your convolved time vector
            # (Assumes same units; usually ms)
            ramp_w_interp = np.interp(t, ramp_t, ramp_w)

            print("\n [DEBUG] RAMP SANITY CHECK")
            print("Requested w_start/w_end:", nn.w_start, nn.w_end)
            print("ramp_t min/max:", float(np.min(ramp_t)), float(np.max(ramp_t)))
            print("ramp_w min/max:", float(np.min(ramp_w)), float(np.max(ramp_w)))
            print("t min/max:", float(np.min(t)), float(np.max(t)))
            print("interp min/max:", float(np.min(ramp_w_interp)), float(np.max(ramp_w_interp)))
            print("last 5 ramp_w:", ramp_w[-5:])


            # ----- FIGURE: 4 rows x 2 cols -----
            fig = plt.figure(figsize=(16, 13), constrained_layout=True)
            gs = gridspec.GridSpec(
                4, 2,
                height_ratios=[1.2, 1.0, 1.2, 0.7],  # RG, V0d, MNP, Weight
                figure=fig
            )

            fig.suptitle(f"{label} | ONLINE RAMP | {ramp_weight_name}", fontsize=14)

            # =========================
            # ROW 1: RG
            # =========================
            ax_rg_left  = fig.add_subplot(gs[0, 0])
            ax_rg_right = fig.add_subplot(gs[0, 1])

            ax_rg_left.plot(t, rg1_convolved * rg1_scale)
            ax_rg_left.plot(t, rg2_convolved * rg2_scale)
            ax_rg_left.set_title("RG Ipsilateral")
            ax_rg_left.legend(["RG_F ipsi", "RG_E ipsi"], fontsize="xx-small")

            ax_rg_right.plot(t, contra_rg1_convolved * contra_rg1_scale, linestyle="--", alpha=0.6)
            ax_rg_right.plot(t, contra_rg2_convolved * contra_rg2_scale, linestyle="--", alpha=0.6)
            ax_rg_right.set_title("RG Contralateral")
            ax_rg_right.legend(["RG_F contra", "RG_E contra"], fontsize="xx-small")

            # =========================
            # ROW 2: V0d (plain lines, no colorbar)
            # =========================
            ax_v0d_left  = fig.add_subplot(gs[1, 0])
            ax_v0d_right = fig.add_subplot(gs[1, 1])

            ax_v0d_left.plot(t, v0d_convolved * v0d_scale)
            ax_v0d_left.set_title("V0d Ipsilateral (Left Inhibitory Output)")
            ax_v0d_left.set_ylabel("Freq (Hz)")

            ax_v0d_right.plot(t, v0d_contra_convolved * v0d_contra_scale)
            ax_v0d_right.set_title("V0d Contralateral (Cross-midline Inhibition)")

            # Keep V0d y-lims comparable between ipsi/contra
            v0d_left_y  = v0d_convolved * v0d_scale
            v0d_right_y = v0d_contra_convolved * v0d_contra_scale
            ymax = np.nanmax([np.nanmax(v0d_left_y), np.nanmax(v0d_right_y)])
            if np.isfinite(ymax) and ymax > 0:
                ax_v0d_left.set_ylim(0, ymax * 1.1)
                ax_v0d_right.set_ylim(0, ymax * 1.1)

            # =========================
            # ROW 3: MNP
            # =========================
            ax_mnp_left  = fig.add_subplot(gs[2, 0])
            ax_mnp_right = fig.add_subplot(gs[2, 1])

            ax_mnp_left.plot(t, mnp1_convolved * mnp1_scale)
            ax_mnp_left.plot(t, mnp2_convolved * mnp2_scale)
            ax_mnp_left.set_title("MNP Ipsilateral")
            ax_mnp_left.legend(["FLX ipsi", "EXT ipsi"], fontsize="xx-small")

            ax_mnp_right.plot(t, contra_mnp1_convolved * contra_mnp1_scale, linestyle="--", alpha=0.6)
            ax_mnp_right.plot(t, contra_mnp2_convolved * contra_mnp2_scale, linestyle="--", alpha=0.6)
            ax_mnp_right.set_title("MNP Contralateral")
            ax_mnp_right.legend(["FLX contra", "EXT contra"], fontsize="xx-small")

            ax_mnp_left.set_xlabel("Time (ms)")
            ax_mnp_right.set_xlabel("Time (ms)")

            import matplotlib.ticker as mticker

            # ----------------------------------------------------------
            # ROW 4: Weight vs time (spans both columns)
            # ----------------------------------------------------------
            ax_w = fig.add_subplot(gs[3, :])
            ax_w.plot(t, ramp_w_interp)

            ax_w.set_title(f"{ramp_weight_name} (weight over time)")
            ax_w.set_xlabel("Time (ms)")
            ax_w.set_ylabel("Weight")

            w_min = float(np.min(ramp_w_interp))
            w_max = float(np.max(ramp_w_interp))

            # y-lims with padding if constant
            if np.isclose(w_min, w_max):
                pad = 0.5 if w_min == 0 else abs(w_min) * 0.1
                ax_w.set_ylim(w_min - pad, w_max + pad)
            else:
                ax_w.set_ylim(w_min, w_max)

            # More y ticks
            rng = abs(w_max - w_min)
            tick_step = 0.25 if rng <= 3 else 0.5

            start = np.floor(w_min / tick_step) * tick_step
            end   = np.ceil(w_max / tick_step) * tick_step + tick_step
            ax_w.set_yticks(np.arange(start, end, tick_step))

            ax_w.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

             # Save once
            if nn.args["save_results"]:
                plt.savefig(
                    nn.pathFigures + f"/{label}_ONLINE_RAMP_RG_V0D_MNP_WEIGHT.png",
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.close(fig)

    else:
        ramp_t = None
        ramp_w = None



        
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
        np.savetxt(nn.pathFigures + '/output_mnp1.csv',mnp1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + '/output_mnp2.csv',mnp2_convolved_scaled,delimiter=',')    

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


        np.savetxt(nn.pathFigures + f'/{label}output_rg1.csv',rg1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_rg2.csv',rg2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v2b.csv',v2b_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v1.csv',v1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v2a1.csv',v2a1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v2a2.csv',v2a2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v0c1.csv',v0c1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v0c2.csv',v0c2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_1a1.csv',v1a1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_1a2.csv',v1a2_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_rc1.csv',rc1_convolved_scaled,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_rc2.csv',rc2_convolved_scaled,delimiter=',')  
        
    if nn.args['save_results'] and nn.rate_coded_plot == 1 and nn.isf_output == 0:
        # Save population rate output
        np.savetxt(nn.pathFigures + f'/{label}output_mnp1.csv',spike_bins_mnp1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_mnp2.csv',spike_bins_mnp2_true,delimiter=',')
        
    if nn.args['save_results'] and nn.save_all_pops==1 and nn.rate_coded_plot == 1 and nn.isf_output == 0:
        np.savetxt(nn.pathFigures + f'/{label}output_rg1.csv',spike_bins_rg1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_rg2.csv',spike_bins_rg2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v2b.csv',spike_bins_inh_inter1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v1.csv',spike_bins_inh_inter2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v2a1.csv',spike_bins_exc_inter1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v2a2.csv',spike_bins_exc_inter2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v0c1.csv',spike_bins_V0c_1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_v0c2.csv',spike_bins_V0c_2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_1a1.csv',spike_bins_V1a_1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_1a2.csv',spike_bins_V1a_2_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_rc1.csv',spike_bins_rc_1_true,delimiter=',')
        np.savetxt(nn.pathFigures + f'/{label}output_rc2.csv',spike_bins_rc_2_true,delimiter=',')

   # ---------------- METRICS TABLE (CLEAN) ----------------
    existing_metrics = {
        # already computed in cpg_utils/cpg_data_utils:
        "rg_flx_isf_max": rg1_isf_max,
        "rg_ext_isf_max": rg2_isf_max,
        "mnp_flx_isf_max": mnp1_isf_max,
        "mnp_ext_isf_max": mnp2_isf_max,
    }

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

