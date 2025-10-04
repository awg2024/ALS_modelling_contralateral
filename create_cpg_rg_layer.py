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
#import elephant
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


# code we need to change. 
import create_flx_rg as flx_rg  # CHANGED. 
import create_ext_rg as ext_rg  # CHANGED. 
import create_exc_inter_pop as exc # CHANGED. 
import create_inh_inter_pop as inh # CHANGED. 
import create_interneuron_pop as inter # CHANGED 
import create_mnp as mnp # CHANGED. 
import calculate_stability_metrics as calc # CHANGED. 
import plotting_utils as pu

warnings.filterwarnings('ignore')


# ===============================
# CREATE RG POPULATIONS
# ===============================
L_rg1 = flx_rg.create_rg_population()  # Left Flexor
L_rg2 = ext_rg.create_rg_population()  # Left Extensor

R_rg1 = flx_rg.create_rg_population()  # Right Flexor
R_rg2 = ext_rg.create_rg_population()  # Right Extensor

if nn.rgs_connected == 1:
    # Create inhibitory interneuron populations for both sides
    L_inh_V2b = inh.create_inh_inter_population('V2b')  # Left V2b
    L_inh_V1  = inh.create_inh_inter_population('V1')   # Left V1

    R_inh_V2b = inh.create_inh_inter_population('V2b')  # Right V2b
    R_inh_V1  = inh.create_inh_inter_population('V1')   # Right V1

    # ===============================
    # WITHIN-HEMISPHERE CONNECTIONS
    # ===============================
    # ---- LEFT SIDE ----
    # Flexor → Inhibitory
    conn.create_connections(L_rg1.rg_exc_bursting, L_inh_V2b.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(L_rg2.rg_exc_bursting, L_inh_V1.inh_inter_tonic,  'custom_rg_v1')

    # Inhibitory → RG (within Left)
    for src in [L_inh_V2b.inh_inter_tonic]:
        for tgt in [L_rg2.rg_exc_bursting, L_rg2.rg_exc_tonic, L_rg2.rg_inh_bursting, L_rg2.rg_inh_tonic]:
            conn.create_connections(src, tgt, 'custom_v2b_rg')

    for src in [L_inh_V1.inh_inter_tonic]:
        for tgt in [L_rg1.rg_exc_bursting, L_rg1.rg_exc_tonic, L_rg1.rg_inh_bursting, L_rg1.rg_inh_tonic]:
            conn.create_connections(src, tgt, 'custom_v1_rg')

    # Excitatory RG cross-coupling (Left Flexor ↔ Left Extensor)
    for src in [L_rg1.rg_exc_bursting, L_rg1.rg_exc_tonic]:
        for tgt in [L_rg2.rg_exc_bursting, L_rg2.rg_exc_tonic]:
            conn.create_connections(src, tgt, 'custom_rg_rg')
    for src in [L_rg2.rg_exc_bursting, L_rg2.rg_exc_tonic]:
        for tgt in [L_rg1.rg_exc_bursting, L_rg1.rg_exc_tonic]:
            conn.create_connections(src, tgt, 'custom_rg_rg')

    # ---- RIGHT SIDE ----
    conn.create_connections(R_rg1.rg_exc_bursting, R_inh_V2b.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(R_rg2.rg_exc_bursting, R_inh_V1.inh_inter_tonic,  'custom_rg_v1')

    for src in [R_inh_V2b.inh_inter_tonic]:
        for tgt in [R_rg2.rg_exc_bursting, R_rg2.rg_exc_tonic, R_rg2.rg_inh_bursting, R_rg2.rg_inh_tonic]:
            conn.create_connections(src, tgt, 'custom_v2b_rg')

    for src in [R_inh_V1.inh_inter_tonic]:
        for tgt in [R_rg1.rg_exc_bursting, R_rg1.rg_exc_tonic, R_rg1.rg_inh_bursting, R_rg1.rg_inh_tonic]:
            conn.create_connections(src, tgt, 'custom_v1_rg')

    for src in [R_rg1.rg_exc_bursting, R_rg1.rg_exc_tonic]:
        for tgt in [R_rg2.rg_exc_bursting, R_rg2.rg_exc_tonic]:
            conn.create_connections(src, tgt, 'custom_rg_rg')
    for src in [R_rg2.rg_exc_bursting, R_rg2.rg_exc_tonic]:
        for tgt in [R_rg1.rg_exc_bursting, R_rg1.rg_exc_tonic]:
            conn.create_connections(src, tgt, 'custom_rg_rg')


print("Seed#: ", nn.rng_seed)
print("RG Flx: # exc (bursting, tonic): ", nn.flx_exc_bursting_count, nn.flx_exc_tonic_count,
      "; # inh(bursting, tonic): ", nn.flx_inh_bursting_count, nn.flx_inh_tonic_count)
print("RG Ext: # exc (bursting, tonic): ", nn.ext_exc_bursting_count, nn.ext_exc_tonic_count,
      "; # inh(bursting, tonic): ", nn.ext_inh_bursting_count, nn.ext_inh_tonic_count)
print("V2b/V1: # inh (bursting): ", nn.num_inh_inter_bursting_v2b, nn.num_inh_inter_bursting_v1,
      "; (tonic): ", nn.num_inh_inter_tonic_v2b, nn.num_inh_inter_tonic_v1)


# building two LEFT and RIGHT RGs together. 

init_time=50
nest.Simulate(init_time)

num_steps = int(nn.sim_time/nn.time_resolution)
t_start = time.perf_counter()
for i in range(int(num_steps/10)-init_time):	
    nest.Simulate(nn.time_resolution*10)
    print("t = " + str(nest.biological_time),end="\r")        
                
t_stop = time.perf_counter()    
print('Simulation completed. It took ',round(t_stop-t_start,2),' seconds.')


# LEFT network
pu.plot_graphs(L_rg1, L_rg2, nn, popfunc, conn, calc, label="Left")

# RIGHT network
pu.plot_graphs(R_rg1, R_rg2, nn, popfunc, conn, calc, label="Right")