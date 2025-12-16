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
              label):

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

    #Read spike data - V2a excitatory interneurons
    senders_exc_inter_tonic1,spiketimes_exc_inter_tonic1 = popfunc.read_spike_data(exc1.spike_detector_exc_inter_tonic)
    senders_exc_inter_tonic2,spiketimes_exc_inter_tonic2 = popfunc.read_spike_data(exc2.spike_detector_exc_inter_tonic)

    # --- CONTRALATERAL V2a POPULATIONS ---
    senders_contra_exc_inter_tonic1, spiketimes_contra_exc_inter_tonic1 = popfunc.read_spike_data(contra_exc1.spike_detector_exc_inter_tonic)
    senders_contra_exc_inter_tonic2, spiketimes_contra_exc_inter_tonic2 = popfunc.read_spike_data(contra_exc2.spike_detector_exc_inter_tonic)


    # --- DEBUG: Spike summary for V0 populations ---
    def debug_spike_report(name, senders, spiketimes):
        print(f"\n===== DEBUG: {name} =====")

        if spiketimes is None or len(spiketimes) == 0:
            print("No spiketimes array found.")
            return

        try:
            # spiketimes[0] = list of spike lists per neuron
            flat_spikes = sum(len(st) for st in spiketimes[0])
            active_neurons = sum(1 for st in spiketimes[0] if len(st) > 0)
            total_neurons = len(spiketimes[0])
        except Exception as e:
            print("Error while parsing spiketimes:", e)
            print("Raw spiketimes:", spiketimes)
            return

        print(f"Total neurons: {total_neurons}")
        print(f"Active neurons: {active_neurons}")
        print(f"Silent neurons: {total_neurons - active_neurons}")
        print(f"Total spikes: {flat_spikes}")

        if flat_spikes == 0:
            print("⚠️  No spikes detected — population is SILENT.")
        else:
            print("Spiking OK ✔️")


    #Read spike data - interneurons
    senders_V0c_1,spiketimes_V0c_1 = popfunc.read_spike_data(V0c_1.spike_detector)
    senders_V0c_2,spiketimes_V0c_2 = popfunc.read_spike_data(V0c_2.spike_detector)
    senders_V1a_1,spiketimes_V1a_1 = popfunc.read_spike_data(V1a_1.spike_detector)
    senders_V1a_2,spiketimes_V1a_2 = popfunc.read_spike_data(V1a_2.spike_detector)
    senders_rc_1,spiketimes_rc_1 = popfunc.read_spike_data(rc_1.spike_detector)
    senders_rc_2,spiketimes_rc_2 = popfunc.read_spike_data(rc_2.spike_detector)


    senders_V0d_contra, spiketimes_V0d_contra = popfunc.read_spike_data(V0d_contra.spike_detector)
    senders_V0d, spiketimes_V0d = popfunc.read_spike_data(V0d.spike_detector)
    senders_V0v, spiketimes_V0v = popfunc.read_spike_data(V0v.spike_detector)

    senders_V0v_contra, spiketimes_V0v_contra = popfunc.read_spike_data(V0v_contra.spike_detector)

    # --- DEBUG: Spike summary for V0 populations ---
    # ------------------ RUN DEBUGGING ------------------
    debug_spike_report("V0d IPSILATERAL",    senders_V0d,        spiketimes_V0d)
    debug_spike_report("V0d CONTRALATERAL",  senders_V0d_contra, spiketimes_V0d_contra)

    debug_spike_report("V0v IPSILATERAL",    senders_V0v,        spiketimes_V0v)
    debug_spike_report("V0v CONTRALATERAL",  senders_V0v_contra, spiketimes_V0v_contra)





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
        print('Max spike count FLX: ',max(spike_bins_mnp1))
        spike_bins_mnp1 = (spike_bins_mnp1-np.min(spike_bins_mnp1))/(np.max(spike_bins_mnp1)-np.min(spike_bins_mnp1))
        spike_bins_mnp2 = popfunc.rate_code_spikes(nn.num_motor_neurons,spiketimes_mnp2)
        spike_bins_mnp2_true = spike_bins_mnp2
        print('Max spike count EXT: ',max(spike_bins_mnp2))
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

        v2a1_freq, v2a1_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
        v2a2_freq, v2a2_times =popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)

        # --- CONTRALATERAL V2a frequencies ---
        contra_v2a1_freq, contra_v2a1_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic1)
        contra_v2a2_freq, contra_v2a2_times = popfunc.calculate_interspike_frequency(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic2)

        
        v0c1_freq, v0c1_times = popfunc.calculate_interspike_frequency(nn.v0c_pop_size,spiketimes_V0c_1)
        v0c2_freq, v0c2_times = popfunc.calculate_interspike_frequency(nn.v0c_pop_size,spiketimes_V0c_2)

        v0v_freq, v0v_times = popfunc.calculate_interspike_frequency(nn.v0v_pop_size,spiketimes_V0v)
        v0d_freq, v0d_times = popfunc.calculate_interspike_frequency(nn.v0d_pop_size,spiketimes_V0d)

        v0d_contra_freq, v0d_contra_times = popfunc.calculate_interspike_frequency(nn.v0d_pop_size, spiketimes_V0d_contra)
        v0v_contra_freq, v0v_contra_times = popfunc.calculate_interspike_frequency(nn.v0v_pop_size, spiketimes_V0v_contra)


        v1a1_freq, v1a1_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_1)
        v1a2_freq, v1a2_times = popfunc.calculate_interspike_frequency(nn.v1a_pop_size,spiketimes_V1a_2)
        
        rc1_freq, rc1_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_1)
        rc2_freq, rc2_times = popfunc.calculate_interspike_frequency(nn.rc_pop_size,spiketimes_rc_2)
        
        if nn.rgs_connected:
            v2b_freq, v2b_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v2b,spiketimes_inh_inter_tonic1)
            v1_freq, v1_times =popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1,spiketimes_inh_inter_tonic2)
            v1_contra_freq, v1_contra_times = popfunc.calculate_interspike_frequency(nn.num_inh_inter_tonic_v1,spikestimes_inh_inter_tonic2_contra)
     
            # Motor neuron instantaneous spiking frequencies
        mnp1_freq, mnp1_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_mnp1
        )
        mnp2_freq, mnp2_times = popfunc.calculate_interspike_frequency(
            nn.num_motor_neurons, spiketimes_mnp2
        )



        print("\n===  Spiking Frequencies (Hz) ===")
        print(f"RG1 Exc (bursting): {np.nanmean([np.nanmean(f) for f in rgexc1_bursting_freq]):.2f} Hz")
        print(f"RG2 Exc (bursting): {np.nanmean([np.nanmean(f) for f in rgexc2_bursting_freq]):.2f} Hz")

        print(f"RG1 Exc (tonic): {np.nanmean([np.nanmean(f) for f in rgexc1_tonic_freq]):.2f} Hz")
        print(f"RG2 Exc (tonic): {np.nanmean([np.nanmean(f) for f in rgexc2_tonic_freq]):.2f} Hz")

        print(f"RG1 Inh (bursting): {np.nanmean([np.nanmean(f) for f in rginh1_bursting_freq]):.2f} Hz")
        print(f"RG2 Inh (bursting): {np.nanmean([np.nanmean(f) for f in rginh2_bursting_freq]):.2f} Hz")

        
        t_stop = time.perf_counter()    
        print('Calculating ISF complete, taking ',int(t_stop-t_start),' seconds.')
        
        t_start = time.perf_counter()
        #Convolve spike data - RG populations
        
        rg_exc_tonic_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_exc_tonic_count,spiketimes_exc_tonic1)
        rg_inh_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_inh_bursting_count,spiketimes_inh1)
        rg_inh_tonic_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_inh_tonic_count,spiketimes_inh_tonic1)
        rg_exc_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count, spiketimes_exc1)
    
        rg1_convolved = np.vstack([rg_exc_convolved1,rg_inh_convolved1])
        rg1_convolved = np.vstack([rg1_convolved,rg_exc_tonic_convolved1])
        rg1_convolved = np.vstack([rg1_convolved,rg_inh_tonic_convolved1])
        rg1_convolved = rg1_convolved.mean(axis=0)

        rg_exc_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count,spiketimes_exc2)
        rg_exc_tonic_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_exc_tonic_count,spiketimes_exc_tonic2)
        rg_inh_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_inh_bursting_count,spiketimes_inh2)
        rg_inh_tonic_convolved2, _  = popfunc.convolve_spiking_activity(nn.ext_inh_tonic_count,spiketimes_inh_tonic2)
        rg2_convolved = np.vstack([rg_exc_convolved2,rg_inh_convolved2])
        rg2_convolved = np.vstack([rg2_convolved,rg_exc_tonic_convolved2])
        rg2_convolved = np.vstack([rg2_convolved,rg_inh_tonic_convolved2])
        rg2_convolved = rg2_convolved.mean(axis=0)

        #Convolve spike data - V2a excitatory interneuron populations
        v2a1_convolved, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic1)
        v2a2_convolved, _  = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size,spiketimes_exc_inter_tonic2)

        # --- CONTRALATERAL V2a convolved ---
        v2a_contra1_convolved, _ = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic1)
        v2a_contra2_convolved, _ = popfunc.convolve_spiking_activity(nn.v2a_tonic_pop_size, spiketimes_contra_exc_inter_tonic2)

        # Average contra V2a if needed (match ipsi structure)
        v2a_contra_convolved = np.vstack([v2a_contra1_convolved, v2a_contra2_convolved]).mean(axis=0)

        
        #Convolve spike data - V2a excitatory interneuron populations
        v0c1_convolved, _  = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_1)
        v0c2_convolved, _  = popfunc.convolve_spiking_activity(nn.v0c_pop_size,spiketimes_V0c_2)
        
        v1a1_convolved, _ = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_1)
        v1a2_convolved, _ = popfunc.convolve_spiking_activity(nn.v1a_pop_size,spiketimes_V1a_2)
        
        rc1_convolved, _ = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_1)
        rc2_convolved, _ = popfunc.convolve_spiking_activity(nn.rc_pop_size,spiketimes_rc_2)

        #Convolve spike data - v2a and v0v 

        v0v_convolved, _ = popfunc.convolve_spiking_activity(nn.v0v_pop_size,spiketimes_V0v)

        v0d_convolved, _ = popfunc.convolve_spiking_activity(nn.v0d_pop_size,spiketimes_V0d)

        v0d_contra_convolved, _ = popfunc.convolve_spiking_activity(nn.v0d_pop_size, spiketimes_V0d_contra)

        v0v_contra_convolved, _ = popfunc.convolve_spiking_activity(nn.v0v_pop_size, spiketimes_V0v_contra)


        #Convolve spike data - MNPs
        mnp1_convolved, convolved_time = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp1)
        mnp2_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons,spiketimes_mnp2)

        # --- Convolve CONTRALATERAL RG populations (full symmetry to ipsilateral) ---

        # Excitatory bursting
        contra_rg_exc_burst_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_exc_bursting_count, spiketimes_contra_rg_exc_burst1)
        contra_rg_exc_burst_convolved2, _ = popfunc.convolve_spiking_activity(nn.ext_exc_bursting_count, spiketimes_contra_rg_exc_burst2)

        # Inhibitory bursting
        contra_rg_inh_burst_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_inh_bursting_count, spiketimes_contra_rg_inh_burst1)
        contra_rg_inh_burst_convolved2, _ = popfunc.convolve_spiking_activity(nn.ext_inh_bursting_count, spiketimes_contra_rg_inh_burst2)

        # Excitatory tonic
        contra_rg_exc_tonic_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_exc_tonic_count, spiketimes_contra_rg_exc_tonic1)
        contra_rg_exc_tonic_convolved2, _ = popfunc.convolve_spiking_activity(nn.ext_exc_tonic_count, spiketimes_contra_rg_exc_tonic2)

        # Inhibitory tonic
        contra_rg_inh_tonic_convolved1, _ = popfunc.convolve_spiking_activity(nn.flx_inh_tonic_count, spiketimes_contra_rg_inh_tonic1)
        contra_rg_inh_tonic_convolved2, _ = popfunc.convolve_spiking_activity(nn.ext_inh_tonic_count, spiketimes_contra_rg_inh_tonic2)

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
        contra_mnp1_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, spiketimes_contra_mnp1)
        contra_mnp2_convolved, _ = popfunc.convolve_spiking_activity(nn.num_motor_neurons, spiketimes_contra_mnp2)

                
        # Convolve spike data - inh populations
        if nn.rgs_connected == 1:
            v2b_convolved, _  = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v2b, spiketimes_inh_inter_tonic1)
            v1_convolved, _  = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v1, spiketimes_inh_inter_tonic2)
            v1_contra_convolved, _ = popfunc.convolve_spiking_activity(nn.num_inh_inter_tonic_v1, spikestimes_inh_inter_tonic2_contra)


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

        
        v0c1_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0c1_freq]))
        v0c2_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0c2_freq]))
        v0c1_conv_max = np.nanmax(v0c1_convolved)
        v0c2_conv_max = np.nanmax(v0c2_convolved)
        v0c1_scale = v0c1_isf_max / v0c1_conv_max
        v0c2_scale = v0c2_isf_max / v0c2_conv_max
        
        v0v_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0v_freq]))
        v0d_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0d_freq]))
        v0d_contra_isf_max = np.nanmax(np.array([np.nanmean(neuron_freq) for neuron_freq in v0d_contra_freq]))
        V0v_conv_max = np.nanmax(v0v_convolved)
        V0d_conv_max = np.nanmax(v0d_convolved)
        V0d_contra_conv_max = np.nanmax(v0d_contra_convolved)
        v0v_contra_isf_max = np.nanmax([np.nanmean(f) for f in v0v_contra_freq])

        v0v_contra_scale = v0v_contra_isf_max / np.nanmax(v0v_contra_convolved)
        v0v_scale = v0v_isf_max / V0v_conv_max
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
        
        mnp1_convolved_scaled = mnp1_convolved * mnp1_scale
        mnp2_convolved_scaled = mnp2_convolved * mnp2_scale
        mnp1_convolved_scaled_mean = np.nanmean(mnp1_convolved_scaled)
        mnp2_convolved_scaled_mean = np.nanmean(mnp2_convolved_scaled)
        mnp1_convolved_max_scaled = np.nanmax(mnp1_convolved * mnp1_scale)
        mnp2_convolved_max_scaled = np.nanmax(mnp2_convolved * mnp2_scale)
        
        print('After scaling max firing rate of a Flx MN (Convolved):',round(mnp1_convolved_max_scaled,2),'Ext MN:',round(mnp2_convolved_max_scaled,2))
        print('After scaling mean firing rate of a Flx MN (Convolved):',round(mnp1_convolved_scaled_mean,2),'Ext MN:',round(mnp2_convolved_scaled_mean,2))
        
        mnp1_avg_norm = (mnp1_convolved-np.min(mnp1_convolved))/(np.max(mnp1_convolved)-np.min(mnp1_convolved))
        mnp2_avg_norm = (mnp2_convolved-np.min(mnp2_convolved))/(np.max(mnp2_convolved)-np.min(mnp2_convolved))
        if max(mnp1_avg_norm)>0 and max(mnp2_avg_norm)>0: 
            avg_freq, avg_phase, bd_comparison = calc.analyze_output(mnp1_avg_norm,mnp2_avg_norm,mnp1_convolved_scaled,mnp2_convolved_scaled,'MNP',y_line_bd=0.4,y_line_phase=0.7)
        
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

            # ============================================================
            # === HEATMAPS FOR V0d AND V0v (Separate Ipsilateral/Contra) ===
            # ============================================================

            def plot_population_heatmap(pop_name, pop_convolved, pop_scale, time_vector, neuron_count, save_path=None):
                """
                Plot individual neuron activation over time (heatmap).
                
                Parameters:
                -----------
                pop_name      : str, e.g. 'V0d Ipsilateral'
                pop_convolved : matrix (N x T) convolved firing rate per neuron OR 1D flattened array
                pop_scale     : scale factor (neuron_output_scale)
                time_vector   : 1D array of time points
                neuron_count  : int, number of neurons in population
                save_path     : str, path to save figure (optional)
                """
                
                # Reshape if flattened
                if pop_convolved.ndim == 1:
                    time_steps = len(pop_convolved) // neuron_count
                    pop_convolved = pop_convolved.reshape(neuron_count, time_steps)
                
                # Apply scaling
                pop_convolved_scaled = pop_convolved * pop_scale

                # NO SORTING - keep original neuron order
                pop_sorted = pop_convolved_scaled

                # --- PLOT ---
                plt.figure(figsize=(16, 8))
                plt.imshow(
                    pop_sorted,
                    aspect='auto',
                    extent=[time_vector[0], time_vector[-1], 0, pop_sorted.shape[0]],
                    origin='lower',
                    cmap='plasma'
                )
                plt.colorbar(label='Firing Rate (Hz)')
                plt.xlabel("Time (ms)")
                plt.ylabel("Neuron ID")
                plt.title(f"{pop_name} – Individual Neuron Activation Over Time")

                if save_path is not None:
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")

                plt.close()
    
            if nn.args['low_locomotion_v0d_left']:
                plot_population_heatmap(
                    "V0d Ipsilateral",
                    v0d_convolved,
                    v0d_scale,
                    t,
                    nn.v0d_pop_size,  # ← Add neuron count here
                    save_path=f"{nn.pathFigures}/{label}_heatmap_V0d_ipsi.png"
                )


            if nn.args['low_locomotion_v0d_right']:
                plot_population_heatmap(
                    "V0d Contralateral",
                    v0d_contra_convolved,
                    v0d_scae,
                    t,
                    nn.v0d_pop_size,  # ← Same neuron count
                    save_path=f"{nn.pathFigures}/{label}_heatmap_V0d_contra.png"
                )

            if nn.args['low_locomotion_v0v_left']:
                plot_population_heatmap(
                    "V0v Ipsilateral",
                    v0v_convolved,
                    v0v_scale,
                    t,
                    nn.v0v_pop_size,  # ← Different neuron count
                    save_path=f"{nn.pathFigures}/{label}_heatmap_V0v_ipsi.png"
                )

            if nn.args['low_locomotion_v0v_right']:
                plot_population_heatmap(
                    "V0v Contralateral",
                    v0v_contra_convolved,
                    v0v_scale,
                    t,
                    nn.v0v_pop_size,  # ← Same neuron count
                    save_path=f"{nn.pathFigures}/{label}_heatmap_V0v_contra.png"
                )

        
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