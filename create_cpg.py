#!/usr/bin/env python
import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import csv 
import os
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
import create_commissural_pop as commissural

import create_mnp as mnp
import calculate_stability_metrics as calc
import send_receive_feedback as interface_fb
from cpg_data_utils import cpg_utils

warnings.filterwarnings('ignore')

# Capture command-line arguments
if len(sys.argv) != 3:
    print("Usage: python3 create_cpg.py <seed> <test_name>")
    sys.exit(1)

SEED = sys.argv[1]
TEST_NAME = sys.argv[2]


#Create neuron populations - NEST
# =====================================
# LEFT POPULATION 

# Left RG flexor 
L_rg1 = flx_rg.create_rg_population()

# Left RG extensor 
L_rg2 = ext_rg.create_rg_population()

L_exc1 = exc.create_exc_inter_population("V2a_L") # exc1 = flx v2a tonic 
L_exc2 = exc.create_exc_inter_population("V2a_L") # exc2 = ext v2a tonic 

# exc1 = flx v2a burst 
L_exc1_burst = inter.interneuron_population()

# Left v1a flexor and extensor 
L_V1a_1 = inter.interneuron_population()
L_V1a_2 = inter.interneuron_population()

L_rc_1 = inter.interneuron_population()
L_rc_2 = inter.interneuron_population()

L_mnp1 = mnp.mnp()
L_mnp2 = mnp.mnp()

L_fb = interface_fb.feedback()

# commissural 
L_V0C_1 = commissural.commissural_population()
L_V0C_2 = commissural.commissural_population()

L_V0D = commissural.commissural_population()

L_V3_Flx = commissural.commissural_population()
L_V3_Ext = commissural.commissural_population()
# =====================================

# RIGHTPOPULATION 
R_rg1 = flx_rg.create_rg_population()
R_rg2 = ext_rg.create_rg_population()

# exc1 = flx v2a tonic 
# exc2 = ext v2a tonic 
R_exc1 = exc.create_exc_inter_population("V2a_R")
R_exc2 = exc.create_exc_inter_population("V2a_R")

# exc1 = flx v2a burst 
# exc2 = ext v2a burst 
R_exc1_burst = inter.interneuron_population()


R_V1a_1 = inter.interneuron_population()
R_V1a_2 = inter.interneuron_population()

R_rc_1 = inter.interneuron_population()
R_rc_2 = inter.interneuron_population()

R_mnp1 = mnp.mnp()
R_mnp2 = mnp.mnp()

R_fb = interface_fb.feedback()

# commissural 
R_V0C_1 = commissural.commissural_population()
R_V0C_2 = commissural.commissural_population()

R_V0D = commissural.commissural_population()

R_V3_Flx = commissural.commissural_population()
R_V3_Ext = commissural.commissural_population()
# =====================================


# CONFIGURING LEFT CONNECTIONS CPG. 
if nn.remove_descending_drive==0:

    # V0c_1 = FLx, V0c_2 = Ext 
    L_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_L,input_type='descending')
    L_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_L,input_type='descending')
    
    L_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_L,input_type='sensory_feedback')
    L_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_L,input_type='sensory_feedback')

    L_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='tonic', pop_size=nn.v0d_pop_size_L, input_type='descending_inh_L') 
    L_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='bursting', pop_size=nn.v0d_pop_size_L, input_type='descending_inh_L') 

    L_V3_Flx.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='descending_exc_v3_L') 
    L_V3_Ext.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='descending_exc_v3_L')
    

elif nn.remove_descending_drive==1:

    L_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_L,input_type='none')
    L_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_L,input_type='none')
    
    L_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_L,input_type='none')
    L_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_L,input_type='none')

    L_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='tonic', pop_size=nn.v0d_pop_size_L, input_type='none') 
    L_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='bursting', pop_size=nn.v0d_pop_size_L, input_type='none') 

    L_V3_Flx.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='none') 
    L_V3_Ext.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='none')
    

if nn.slow_syn_bias == 'flx':
    print('Slow synaptic dynamics applied to Flexor side only.')
    L_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_L,input_type='none')
    L_rc_2.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_L,input_type='none')
    L_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled',side="LEFT")
    L_mnp2.create_mnp(pop_type='mnp_slow_syn_disabled',side="LEFT")

elif nn.slow_syn_bias == 'ext':
    
    print('Slow synaptic dynamics applied to Extensor side only.')
    L_rc_1.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_L,input_type='none')
    L_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_L,input_type='none')
    L_mnp1.create_mnp(pop_type='mnp_slow_syn_disabled',side="LEFT")
    L_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled',side="LEFT")

else:
    # default setting slow synaptic dynamics... slow of locomotion...

    print('Slow synaptic dynamics applied to Flexor and Extensor.')
    L_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_L,input_type='none')
    L_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_L,input_type='none')
    L_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled',side="LEFT")
    L_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled',side="LEFT")


#Connect rg neurons to V2a excitatory interneuron populations
# FOR EXC... 
# 1 - FLX
# 2 - EXT
conn.create_connections(L_rg1.rg_exc_bursting, L_exc1.exc_inter_tonic,'custom_rg_v2a_L')
conn.create_connections(L_rg1.rg_exc_tonic, L_exc1.exc_inter_tonic,'custom_rg_v2a_L')

conn.create_connections(L_rg2.rg_exc_bursting, L_exc2.exc_inter_tonic,'custom_rg_v2a_L')
conn.create_connections(L_rg2.rg_exc_tonic, L_exc2.exc_inter_tonic,'custom_rg_v2a_L')

#Connect V2a excitatory interneuron populations to motor neurons
conn.create_connections(L_exc1.exc_inter_tonic, L_mnp1.motor_neuron_pop,'custom_v2a_mn_L')
conn.create_connections(L_exc2.exc_inter_tonic, L_mnp2.motor_neuron_pop,'custom_v2a_mn_L')

#Connect rg neurons to V1a excitatory interneuron populations
conn.create_connections(L_rg1.rg_exc_bursting, L_V1a_2.interneuron_pop,'custom_rg_v1a_L')
conn.create_connections(L_rg1.rg_exc_tonic, L_V1a_2.interneuron_pop,'custom_rg_v1a_L')

conn.create_connections(L_rg2.rg_exc_bursting, L_V1a_1.interneuron_pop,'custom_rg_v1a_L')
conn.create_connections(L_rg2.rg_exc_tonic, L_V1a_1.interneuron_pop,'custom_rg_v1a_L')

#Connect rg neurons to V0c interneurons
conn.create_connections(L_rg1.rg_exc_bursting, L_V0C_1.v0c_tonic,'custom_rg_v0c_L')
conn.create_connections(L_rg1.rg_exc_tonic, L_V0C_1.v0c_tonic,'custom_rg_v0c_L')

conn.create_connections(L_rg2.rg_exc_bursting, L_V0C_2.v0c_tonic,'custom_rg_v0c_L')
conn.create_connections(L_rg2.rg_exc_tonic, L_V0C_2.v0c_tonic,'custom_rg_v0c_L')

#Connect V0c to motor neurons
conn.create_connections(L_V0C_1.v0c_tonic, L_mnp1.motor_neuron_pop,'custom_v0c_mn_L')
conn.create_connections(L_V0C_2.v0c_tonic, L_mnp2.motor_neuron_pop,'custom_v0c_mn_L')

#Connect V1a interneurons to contralateral V1a interneurons
conn.create_connections(L_V1a_2.interneuron_pop, L_V1a_1.interneuron_pop,'custom_v1a_v1a_L')
conn.create_connections(L_V1a_1.interneuron_pop, L_V1a_2.interneuron_pop,'custom_v1a_v1a_L')

#Connect V1a to motor neurons
conn.create_connections(L_V1a_1.interneuron_pop, L_mnp1.motor_neuron_pop,'custom_v1a_mn_L')
conn.create_connections(L_V1a_2.interneuron_pop, L_mnp2.motor_neuron_pop,'custom_v1a_mn_L')

#Connect RC interneurons to V1a interneurons
conn.create_connections(L_rc_1.interneuron_pop, L_V1a_2.interneuron_pop,'custom_rc_v1a_L')
conn.create_connections(L_rc_2.interneuron_pop, L_V1a_1.interneuron_pop,'custom_rc_v1a_L')

#Connect RC interneurons to contralateral RC interneurons
conn.create_connections(L_rc_1.interneuron_pop, L_rc_2.interneuron_pop,'custom_rc_rc_L')
conn.create_connections(L_rc_2.interneuron_pop, L_rc_1.interneuron_pop,'custom_rc_rc_L')

#Connect RC interneurons to motor neurons
conn.create_connections(L_rc_1.interneuron_pop, L_mnp1.motor_neuron_pop,'custom_rc_mn_L')
conn.create_connections(L_rc_2.interneuron_pop, L_mnp2.motor_neuron_pop,'custom_rc_mn_L')
conn.create_connections(L_mnp1.motor_neuron_pop, L_rc_1.interneuron_pop,'custom_mn_rc')
conn.create_connections(L_mnp2.motor_neuron_pop, L_rc_2.interneuron_pop,'custom_mn_rc')

if nn.rgs_connected == 1:
    L_inh1 = inh.create_inh_inter_population('V2b_L')  # V2b
    L_inh2 = inh.create_inh_inter_population('V1_L')  # V1

    # Connect excitatory rg neurons to V1/V2b inhibitory populations
    conn.create_connections(L_rg1.rg_exc_bursting, L_inh1.inh_inter_tonic, 'custom_rg_v2b_L')
    #conn.create_connections(L_rg1.rg_exc_tonic, L_inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(L_rg2.rg_exc_bursting, L_inh2.inh_inter_tonic, 'custom_rg_v1_L')
    #conn.create_connections(L_rg2.rg_exc_tonic, L_inh2.inh_inter_tonic, 'custom_rg_v1')

    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_exc_bursting,'custom_v2b_rg_L')
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_exc_tonic,'custom_v2b_rg_L')
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_inh_bursting,'custom_v2b_rg_L')
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_inh_tonic,'custom_v2b_rg_L')

    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_exc_bursting, "custom_v1_rg_L")
    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_exc_tonic,    "custom_v1_rg_L")
    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_inh_bursting, "custom_v1_rg_L")
    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_inh_tonic,    "custom_v1_rg_L")

    #Connect V1/V2b inhibitory populations to V2a
    conn.create_connections(L_inh1.inh_inter_tonic, L_exc2.exc_inter_tonic,'custom_v2b_v2a_L')
    conn.create_connections(L_inh2.inh_inter_tonic, L_exc1.exc_inter_tonic,'custom_v1_v2a_L')

    if nn.v1v2b_mn_connected==1:
        #Connect V1/V2b inhibitory populations to motor neurons
        conn.create_connections(L_inh1.inh_inter_tonic, L_mnp2.motor_neuron_pop,'custom_v2b_mn_L')
        conn.create_connections(L_inh2.inh_inter_tonic, L_mnp1.motor_neuron_pop,'custom_v1_mn_L')
   
    # connect excitatory rg neurons
    conn.create_connections(L_rg1.rg_exc_bursting, L_rg2.rg_exc_bursting,'custom_rg_rg_L')
    conn.create_connections(L_rg1.rg_exc_bursting, L_rg2.rg_exc_tonic,'custom_rg_rg_L')
    conn.create_connections(L_rg1.rg_exc_tonic, L_rg2.rg_exc_bursting,'custom_rg_rg_L')
    conn.create_connections(L_rg1.rg_exc_tonic, L_rg2.rg_exc_tonic,'custom_rg_rg_L')

    conn.create_connections(L_rg2.rg_exc_bursting, L_rg1.rg_exc_bursting,'custom_rg_rg_L')
    conn.create_connections(L_rg2.rg_exc_bursting, L_rg1.rg_exc_tonic,'custom_rg_rg_L')
    conn.create_connections(L_rg2.rg_exc_tonic, L_rg1.rg_exc_bursting,'custom_rg_rg_L')
    conn.create_connections(L_rg2.rg_exc_tonic, L_rg1.rg_exc_tonic,'custom_rg_rg_L')

  
    # V0D CONTRALATERAL  
    if nn.low_locomotion_v0d_left == 1: 
        print("Creating Connections for LEFT V0d - LOW LOCOMOTION...")

        # ASYMMETRIC V0D LOW LOCOMOTION CONNECTION 
        conn.create_connections(L_rg1.rg_exc_bursting, L_V0D.v0d_bursting, 'custom_rg_v0d_L') # RG_F → V0d_bursting - custom_rg_v0d generic connection we are applying across all connections  
        conn.create_connections(L_V0D.v0d_bursting, R_rg1.rg_exc_bursting, 'custom_v0d_rg_inh_L') # V0d_bursting → RG_F CONTRA (inhibitory, correct) - BURSTING 
        # popfunc.print_last_conn_stats(conn, "custom_v0d_rg_inh_L")

  # ================================ END OF LEFT SIDE CPG CODE ==========================================================

if nn.remove_descending_drive==0:
    
    R_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_R,input_type='descending')
    R_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_R,input_type='descending')
    
    R_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_R,input_type='sensory_feedback')
    R_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_R,input_type='sensory_feedback')

    # create contralateral interneuron populations - BURSTING AND TONIC SUBTYPES 
    R_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='tonic', pop_size=nn.v0d_pop_size_R, input_type='descending_inh_R') 
    R_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='bursting', pop_size=nn.v0d_pop_size_R, input_type='descending_inh_R') 

    R_V3_Flx.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='descending_exc_v3_R') 
    R_V3_Ext.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='descending_exc_v3_R') 

elif nn.remove_descending_drive==1:
    
    R_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_R,input_type='none')
    R_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size_R,input_type='none')
   
    R_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_R,input_type='none')
    R_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size_R,input_type='none')

       # create contralateral interneuron populations - BURSTING AND TONIC SUBTYPES 
    R_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='tonic', pop_size=nn.v0d_pop_size_R, input_type='none') 
    R_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='bursting', pop_size=nn.v0d_pop_size_R, input_type='none') 

    R_V3_Flx.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='none') 
    R_V3_Ext.create_commissural_population(pop_type='V3', self_connection='none', firing_behavior='bursting', pop_size=nn.v3_pop_size, input_type='none') 

if nn.slow_syn_bias == 'flx':
    print('Slow synaptic dynamics applied to Flexor side only.')
    R_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_R,input_type='none')
    R_rc_2.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_R,input_type='none')
    R_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled',side="RIGHT")
    R_mnp2.create_mnp(pop_type='mnp_slow_syn_disabled',side="RIGHT")
elif nn.slow_syn_bias == 'ext':
    print('Slow synaptic dynamics applied to Extensor side only.')
    R_rc_1.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_R,input_type='none')
    R_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_R,input_type='none')
    R_mnp1.create_mnp(pop_type='mnp_slow_syn_disabled',side="RIGHT")
    R_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled',side="RIGHT")
else:
    print('Slow synaptic dynamics applied to Flexor and Extensor.')
    R_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_R,input_type='none')
    R_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size_R,input_type='none')
    R_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled',side="RIGHT")
    R_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled',side="RIGHT")


if nn.low_locomotion_v3_left == 1: 

    print("Creating connections for LEFT V3 - Low Locomotion")
    
    # flx-flx
    conn.create_connections(L_rg1.rg_exc_bursting, L_V3_Flx.v3_bursting, 'custom_rg_v3_L')        # ipsilateral connection with the mnp1 
    conn.create_connections(L_V3_Flx.v3_bursting, R_rg1.rg_exc_bursting, 'custom_v3_rg_L')   # custom_v3_rg_L Commissural conection 

    # ext-ext
    conn.create_connections(L_rg2.rg_exc_bursting, L_V3_Ext.v3_bursting, 'custom_rg_v3_L')
    conn.create_connections(L_V3_Ext.v3_bursting, R_rg2.rg_exc_bursting, 'custom_v3_rg_L') # custom_v3_rg_L
    

#Connect rg neurons to V2a excitatory interneuron populations
conn.create_connections(R_rg1.rg_exc_bursting, R_exc1.exc_inter_tonic,'custom_rg_v2a_R')
conn.create_connections(R_rg1.rg_exc_tonic, R_exc1.exc_inter_tonic,'custom_rg_v2a_R')

conn.create_connections(R_rg2.rg_exc_bursting, R_exc2.exc_inter_tonic,'custom_rg_v2a_R')
conn.create_connections(R_rg2.rg_exc_tonic, R_exc2.exc_inter_tonic,'custom_rg_v2a_R')

#Connect V2a excitatory interneuron populations to motor neurons
conn.create_connections(R_exc1.exc_inter_tonic, R_mnp1.motor_neuron_pop,'custom_v2a_mn_R')
conn.create_connections(R_exc2.exc_inter_tonic, R_mnp2.motor_neuron_pop,'custom_v2a_mn_R')

#Connect rg neurons to V1a excitatory interneuron populations
conn.create_connections(R_rg1.rg_exc_bursting, R_V1a_2.interneuron_pop,'custom_rg_v1a_R')
conn.create_connections(R_rg1.rg_exc_tonic, R_V1a_2.interneuron_pop,'custom_rg_v1a_R')

conn.create_connections(R_rg2.rg_exc_bursting, R_V1a_1.interneuron_pop,'custom_rg_v1a_R')
conn.create_connections(R_rg2.rg_exc_tonic, R_V1a_1.interneuron_pop,'custom_rg_v1a_R')

#Connect rg neurons to V0c interneurons
conn.create_connections(R_rg1.rg_exc_bursting, R_V0C_1.v0c_tonic,'custom_rg_v0c_R')
conn.create_connections(R_rg1.rg_exc_tonic, R_V0C_1.v0c_tonic,'custom_rg_v0c_R')

conn.create_connections(R_rg2.rg_exc_bursting, R_V0C_2.v0c_tonic,'custom_rg_v0c_R')
conn.create_connections(R_rg2.rg_exc_tonic, R_V0C_2.v0c_tonic,'custom_rg_v0c_R')

#Connect V0c to motor neurons
conn.create_connections(R_V0C_1.v0c_tonic, R_mnp1.motor_neuron_pop,'custom_v0c_mn_R')
conn.create_connections(R_V0C_2.v0c_tonic, R_mnp2.motor_neuron_pop,'custom_v0c_mn_R')

# Connect V1a flexor/extensor subpops within the same side
conn.create_connections(R_V1a_2.interneuron_pop, R_V1a_1.interneuron_pop,'custom_v1a_v1a_R')
conn.create_connections(R_V1a_1.interneuron_pop, R_V1a_2.interneuron_pop,'custom_v1a_v1a_R')

#Connect V1a to motor neurons
conn.create_connections(R_V1a_1.interneuron_pop, R_mnp1.motor_neuron_pop,'custom_v1a_mn_R')
conn.create_connections(R_V1a_2.interneuron_pop, R_mnp2.motor_neuron_pop,'custom_v1a_mn_R')

#Connect RC interneurons to V1a interneurons
conn.create_connections(R_rc_1.interneuron_pop, R_V1a_2.interneuron_pop,'custom_rc_v1a_R')
conn.create_connections(R_rc_2.interneuron_pop, R_V1a_1.interneuron_pop,'custom_rc_v1a_R')

#Connect RC interneurons to contralateral RC interneurons
conn.create_connections(R_rc_1.interneuron_pop, R_rc_2.interneuron_pop,'custom_rc_rc_R')
conn.create_connections(R_rc_2.interneuron_pop, R_rc_1.interneuron_pop,'custom_rc_rc_R')

#Connect RC interneurons to motor neurons
conn.create_connections(R_rc_1.interneuron_pop, R_mnp1.motor_neuron_pop,'custom_rc_mn_R')
conn.create_connections(R_rc_2.interneuron_pop, R_mnp2.motor_neuron_pop,'custom_rc_mn_R')
conn.create_connections(R_mnp1.motor_neuron_pop, R_rc_1.interneuron_pop,'custom_mn_rc')
conn.create_connections(R_mnp2.motor_neuron_pop, R_rc_2.interneuron_pop,'custom_mn_rc')

if nn.rgs_connected == 1:
    R_inh1 = inh.create_inh_inter_population('V2b_R')  # V2b
    R_inh2 = inh.create_inh_inter_population('V1_R')  # V1

    # Connect excitatory rg neurons to V1/V2b inhibitory populations
    conn.create_connections(R_rg1.rg_exc_bursting, R_inh1.inh_inter_tonic, 'custom_rg_v2b_R')

    #conn.create_connections(R_rg1.rg_exc_tonic, R_inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(R_rg2.rg_exc_bursting, R_inh2.inh_inter_tonic, 'custom_rg_v1_R')
    #conn.create_connections(R_rg2.rg_exc_tonic, R_inh2.inh_inter_tonic, 'custom_rg_v1')

    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_exc_bursting,'custom_v2b_rg_R')
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_exc_tonic,'custom_v2b_rg_R')
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_inh_bursting,'custom_v2b_rg_R')
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_inh_tonic,'custom_v2b_rg_R')

    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_exc_bursting, "custom_v1_rg_R")
    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_exc_tonic,    "custom_v1_rg_R")
    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_inh_bursting, "custom_v1_rg_R")
    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_inh_tonic,    "custom_v1_rg_R")

    #Connect V1/V2b inhibitory populations to V2a
    conn.create_connections(R_inh1.inh_inter_tonic, R_exc2.exc_inter_tonic,'custom_v2b_v2a_R')
    conn.create_connections(R_inh2.inh_inter_tonic, R_exc1.exc_inter_tonic,'custom_v1_v2a_R')

    if nn.v1v2b_mn_connected==1:
        #Connect V1/V2b inhibitory populations to motor neurons
        conn.create_connections(R_inh1.inh_inter_tonic, R_mnp2.motor_neuron_pop,'custom_v2b_mn_R')
        conn.create_connections(R_inh2.inh_inter_tonic, R_mnp1.motor_neuron_pop,'custom_v1_mn_R')

    #Connect excitatory rg neurons
    conn.create_connections(R_rg1.rg_exc_bursting, R_rg2.rg_exc_bursting,'custom_rg_rg_R')
    conn.create_connections(R_rg1.rg_exc_bursting, R_rg2.rg_exc_tonic,'custom_rg_rg_R')
    conn.create_connections(R_rg1.rg_exc_tonic, R_rg2.rg_exc_bursting,'custom_rg_rg_R')
    conn.create_connections(R_rg1.rg_exc_tonic, R_rg2.rg_exc_tonic,'custom_rg_rg_R')

    conn.create_connections(R_rg2.rg_exc_bursting, R_rg1.rg_exc_bursting,'custom_rg_rg_R')
    conn.create_connections(R_rg2.rg_exc_bursting, R_rg1.rg_exc_tonic,'custom_rg_rg_R')
    conn.create_connections(R_rg2.rg_exc_tonic, R_rg1.rg_exc_bursting,'custom_rg_rg_R')
    conn.create_connections(R_rg2.rg_exc_tonic, R_rg1.rg_exc_tonic,'custom_rg_rg_R')

    if nn.low_locomotion_v0d_right == 1: 
        
        print("Creating Connections for RIGHT V0d - LOW LOCOMOTION... ")

        conn.create_connections(R_rg1.rg_exc_bursting, R_V0D.v0d_bursting, 'custom_rg_v0d_R') # low locomotion burst ipsilateral connection - RG 1 FLEX  to R_V0D excitatory connection. 
        conn.create_connections(R_V0D.v0d_bursting, L_rg1.rg_exc_bursting, 'custom_v0d_rg_inh_R') # bursting v0d to contralateral rg bursting - INHIBITORY CONNECTION
        # popfunc.print_last_conn_stats(conn, "custom_v0d_rg_inh_R")
    
    if nn.low_locomotion_v3_right == 1: 

        print("Creating connections for RIGHT V3 - Low Locomotion")
        
        # flx-flx
        conn.create_connections(R_rg1.rg_exc_bursting, R_V3_Flx.v3_bursting, 'custom_rg_v3_R')        # ipsilateral connection with the mnp1 
        conn.create_connections(R_V3_Flx.v3_bursting, L_rg1.rg_exc_bursting, 'custom_v3_rg_R')   # custom_v3_mnp_R Commissural conection 

        # ext-ext 
        conn.create_connections(R_rg2.rg_exc_bursting, R_V3_Ext.v3_bursting, 'custom_rg_v3_R')
        conn.create_connections(R_V3_Ext.v3_bursting, R_rg2.rg_exc_bursting, 'custom_v3_rg_R') # custom_v3_mnp_R


    # ================================ END OF RIGHT SIDE CPG DEFINITIONS & END OF CONN.CREATE_CONNECTIONS ==========================================================

    ##  =============================== ONLINE WEIGHT RAMP =================================
    if nn.online_ramp_experiment == 1 and nn.online_ramp_weight_experiment == 1: 

        print("[INFO] Online Ramp Weight Experiment is ON")
        print("scale_start =", nn.w_start, "scale_end =", nn.w_end)

        ramp_duration = float(nn.ramp_duration)  # ms

        online_ramp_log = {
            "time": [],
            "mult": [],
            "mean_weight": [],
            "weight": [],
            "collapse": [],          # 0/1 each step
            "collapse_time": None,   # first time collapse is detected
            "collapse_mult": None,   # multiplier at collapse
            "collapse_weight": None,
            "plateaus": [] # mean weight at collapse
        }

        # collapse condition intialisation 
        collapse_detector = R_rg1.spike_detector_rg_exc_bursting
        collapse_n_neurons = nn.flx_exc_bursting_count
        collapse_threshold_hz = 4.0      # per-neuron mean Hz threshold (tune)
        collapse_window_ms = 200.0       # temporal precision of detection
        collapse_detected = False

        specified_weight = nn.online_ramp_weight  

        conns_list = conn.synapses.get(specified_weight, [])
        if not conns_list:
            raise RuntimeError(f"No synapses recorded for key: {specified_weight}")

        # flatten list of SynapseCollections into ONE SynapseCollection
        connections = conns_list[0]
        for c in conns_list[1:]:
            connections += c

        print(f"[INFO] Ramp target: {specified_weight}, n_conns={len(connections)}")

        # ---- sanity check: show what these connections actually are ----
        sample = nest.GetStatus(connections[:5], ["source", "target", "weight"])
        print("[DEBUG] sample connections (source,target,weight):", sample)

        # ---- snapshot baseline weights ONCE (preserve distribution + sign) ----
        w0 = np.asarray(nest.GetStatus(connections, "weight"), dtype=float)
        print("[BASELINE weights]")
        print("  mean:", float(np.mean(w0)), "min:", float(np.min(w0)), "max:", float(np.max(w0)))

        # If you want "increase inhibition", w0 should already be negative on average.
        # If mean is positive, you are not really inhibitory overall.
        if np.mean(w0) > 0:
            print("[WARN] Meanç weight is positive. This pathway may not be net-inhibitory.")

        # init sim
        init_time = 50.0
        nest.Simulate(init_time)
        t0 = float(nest.biological_time)

        step_ms = float(nn.time_resolution) * 10.0

        ramp_duration = float(nn.ramp_duration)   # ms
        num_steps = max(1, int(round(ramp_duration / step_ms)))

        t_start = time.perf_counter()

        for _ in range(num_steps):

            nest.Simulate(step_ms)
            t = float(nest.biological_time)

            # Progress through ramp (0 to 1)
            if ramp_duration > 0:
                prog = (t - t0) / ramp_duration
            else:
                prog = 1.0
            prog = max(0.0, min(1.0, prog))

            # Interpret config scaling:
            # 0 → baseline (1×)
            # -3 → 3× inhibition
            start_mult = 1.0
            end_mult   = abs(float(nn.w_end))   # -3 → 3.0

            # Linear ramp of multiplier
            mult = start_mult + (end_mult - start_mult) * prog

            # Apply: preserves sign of inhibitory weights
            w_new = w0 * mult
            nest.SetStatus(connections, [{"weight": float(w)} for w in w_new])

            # Log BOTH multiplier and real mean weight
            online_ramp_log["time"].append(t)
            online_ramp_log["mult"].append(mult)
            mw = float(np.mean(w_new))
            online_ramp_log["mean_weight"].append(mw)
            # Backward-compatible key used by existing plotting utilities
            online_ramp_log["weight"].append(mw)

            # --- MID-RAMP collapse check (rolling window) ---
            t1 = float(nest.biological_time)
            t0w = max(float(t0), t1 - collapse_window_ms)  # don’t go before ramp start

            rate_out = popfunc.window_rates_from_spike_detector(
                collapse_detector,
                collapse_n_neurons,
                t0w,
                t1
            )

            mean_hz = rate_out["mean_hz"]  # per-neuron mean (single number)
            is_collapsed = (mean_hz < collapse_threshold_hz)
            online_ramp_log["collapse"].append(1 if is_collapsed else 0)

            # --- feedback L ---
            num_spikes_flx_L = popfunc.read_recent_spike_data(L_mnp1.spike_detector_motor)
            num_spikes_ext_L = popfunc.read_recent_spike_data(L_mnp2.spike_detector_motor)
            L_fb.send_muscle_activation(num_spikes_flx_L, num_spikes_ext_L)
            flx_1a_L, ext_1a_L, flx_1b_L, ext_1b_L, *_ = L_fb.receive_muscle_afferents()

            if nn.fb_rg_flx: nest.SetStatus(L_rg1.rg_flx_pg, {"rate": flx_1a_L})
            if nn.fb_rg_ext: nest.SetStatus(L_rg2.rg_ext_pg, {"rate": ext_1b_L})
            if nn.fb_v2b:    nest.SetStatus(L_inh1.v2b_pg, {"rate": flx_1a_L})
            if nn.fb_v1:     nest.SetStatus(L_inh2.v1_pg, {"rate": ext_1b_L})

            # --- feedback R ---
            num_spikes_flx_R = popfunc.read_recent_spike_data(R_mnp1.spike_detector_motor)
            num_spikes_ext_R = popfunc.read_recent_spike_data(R_mnp2.spike_detector_motor)
            R_fb.send_muscle_activation(num_spikes_flx_R, num_spikes_ext_R)
            flx_1a_R, ext_1a_R, flx_1b_R, ext_1b_R, *_ = R_fb.receive_muscle_afferents()

            if nn.fb_rg_flx: nest.SetStatus(R_rg1.rg_flx_pg, {"rate": flx_1a_R})
            if nn.fb_rg_ext: nest.SetStatus(R_rg2.rg_ext_pg, {"rate": ext_1b_R})
            if nn.fb_v2b:    nest.SetStatus(R_inh1.v2b_pg, {"rate": flx_1a_R})
            if nn.fb_v1:     nest.SetStatus(R_inh2.v1_pg, {"rate": ext_1b_R})


            print(
                f"t={t:.1f} ms | mult={mult:.2f} | mean weight={np.mean(w_new):.3f}",
                end="\r"
            )

        w_after = np.array(nest.GetStatus(connections, "weight"), dtype=float)
        print("\nAFTER mean:", float(np.mean(w_after)),
            "min:", float(np.min(w_after)),
            "max:", float(np.max(w_after)))

        t_stop = time.perf_counter()
        print(f"[INFO] Online Ramp completed in {t_stop - t_start:.2f} s")

        # ================= CPG UTILS FOR BOTH SIDES =================
        L_label = "LEFT"
        R_label = "RIGHT"

        print("[INFO] Calling LEFT utilis")

        cpg_utils(
            nn, popfunc, conn,
            L_rg1, L_rg2, R_rg1, R_rg2, 
            L_exc1, L_exc2, R_exc1, R_exc2, 
            L_V0C_1, L_V0C_2,
            L_V1a_1, L_V1a_2,
            L_inh1, L_inh2, R_inh2,
            L_rc_1, L_rc_2,
            L_mnp1, L_mnp2, R_mnp1, R_mnp2,
            L_V0D, R_V0D,
            L_label, 
            online_ramp_log, # we passing dict for online experiment. dict =  time & applied weight. 
            nn.online_ramp_weight,
            ramp_type="online_ramp_weight"
        )

        print("[INFO] Calling RIGHT utilis")

        cpg_utils(
            nn, popfunc, conn,
            R_rg1, R_rg2, L_rg1, L_rg2,
            R_exc1, R_exc2, L_exc1, L_exc2,
            R_V0C_1, R_V0C_2,
            R_V1a_1, R_V1a_2,
            R_inh1, R_inh2, L_inh2,
            R_rc_1, R_rc_2,
            R_mnp1, R_mnp2, L_mnp1, L_mnp2,
            R_V0D, L_V0D,
            R_label,
            online_ramp_log, # we passing dict for online experiment. dict =  time & applied weight. 
            nn.online_ramp_weight,
            ramp_type="online_ramp_weight"
            )
        
    # ================= END ONLINE RAMP WEIGHT EXPERIMENT ======================

    ##  =============================== ONLINE DRIVE RAMP ====================================
    if nn.online_ramp_experiment == 1 and nn.online_ramp_drive_experiment == 1:
        
        print("[INFO] Online Ramp Drive Experiment (Using ABSOLUTE I_e Values) is ON")
        t_start = time.perf_counter()

        drive_param = "I_e"
        ramp_duration = float(nn.ramp_duration)

        raw_targets = nn.online_stepwise_drive_targets
        drive_targets = popfunc.resolve_targets_to_nc(
            raw_targets,
            roots={"L_V0D": L_V0D, "R_V0D": R_V0D,
                    "L_V3": L_V3_Flx, "R_V3": R_V3_Flx}
        )

        # ABSOLUTE drives (NOT multipliers)
        drive_start = float(nn.d_start)   # e.g. -180.0
        drive_end   = float(nn.d_end)     # e.g. -240.0

        online_ramp_log = {
            "time": [],
            "drive": [],
            "collapse": [],
            "collapse_time": None,
            "collapse_drive": None,
        }

        # set start drive
        for pop in drive_targets:
            nest.SetStatus(pop, {drive_param: float(drive_start)})

        # burn-in
        nest.Simulate(50.0)
        t0 = float(nest.biological_time)

        step_ms = float(nn.time_resolution) * 10.0
        num_steps = max(1, int(round(ramp_duration / step_ms))) if ramp_duration > 0 else 1

        collapse_detector     = R_rg1.spike_detector_rg_exc_bursting
        collapse_n_neurons    = nn.flx_exc_bursting_count
        
        collapse_threshold_hz = 10.0
        collapse_window_ms    = 200.0
        collapse_detected     = False

        for _ in range(num_steps):

            nest.Simulate(step_ms)
            t = float(nest.biological_time)

            prog = (t - t0) / ramp_duration if ramp_duration > 0 else 1.0
            prog = max(0.0, min(1.0, prog))

            # ABSOLUTE ramp: I_e(t) = start + (end-start)*prog
            x_new = drive_start + (drive_end - drive_start) * prog

            for pop in drive_targets:
                nest.SetStatus(pop, {drive_param: float(x_new)})

            # log
            online_ramp_log["time"].append(t)
            online_ramp_log["drive"].append(float(x_new))

            # collapse check
            t1  = float(nest.biological_time)
            t0w = max(t0, t1 - collapse_window_ms)

            rate_out = popfunc.window_rates_from_spike_detector(
                collapse_detector, collapse_n_neurons, t0w, t1
            )
            mean_hz = float(rate_out["mean_hz"])
            # is_collapsed = (mean_hz < collapse_threshold_hz)
            # online_ramp_log["collapse"].append(1 if is_collapsed else 0)


            # --- feedback L ---
            num_spikes_flx_L = popfunc.read_recent_spike_data(L_mnp1.spike_detector_motor)
            num_spikes_ext_L = popfunc.read_recent_spike_data(L_mnp2.spike_detector_motor)
            L_fb.send_muscle_activation(num_spikes_flx_L, num_spikes_ext_L)
            flx_1a_L, ext_1a_L, flx_1b_L, ext_1b_L, *_ = L_fb.receive_muscle_afferents()

            if nn.fb_rg_flx: nest.SetStatus(L_rg1.rg_flx_pg, {"rate": flx_1a_L})
            if nn.fb_rg_ext: nest.SetStatus(L_rg2.rg_ext_pg, {"rate": ext_1b_L})
            if nn.fb_v2b:    nest.SetStatus(L_inh1.v2b_pg, {"rate": flx_1a_L})
            if nn.fb_v1:     nest.SetStatus(L_inh2.v1_pg, {"rate": ext_1b_L})

            # --- feedback R ---
            num_spikes_flx_R = popfunc.read_recent_spike_data(R_mnp1.spike_detector_motor)
            num_spikes_ext_R = popfunc.read_recent_spike_data(R_mnp2.spike_detector_motor)
            R_fb.send_muscle_activation(num_spikes_flx_R, num_spikes_ext_R)
            flx_1a_R, ext_1a_R, flx_1b_R, ext_1b_R, *_ = R_fb.receive_muscle_afferents()

            if nn.fb_rg_flx: nest.SetStatus(R_rg1.rg_flx_pg, {"rate": flx_1a_R})
            if nn.fb_rg_ext: nest.SetStatus(R_rg2.rg_ext_pg, {"rate": ext_1b_R})
            if nn.fb_v2b:    nest.SetStatus(R_inh1.v2b_pg, {"rate": flx_1a_R})
            if nn.fb_v1:     nest.SetStatus(R_inh2.v1_pg, {"rate": ext_1b_R})

            print(f"t={t:.1f} ms | I_e={x_new:.2f}", end="\r")

        t_stop = time.perf_counter()
        print("\n" + "=" * 70)
        print(f"[INFO] Online Ramp completed in {t_stop - t_start:.2f} s")

        # ================= CPG UTILS FOR BOTH SIDES =================
        L_label = "LEFT"
        R_label = "RIGHT"

        print("[INFO] Calling LEFT utils")
        cpg_utils(
            nn, popfunc, conn,
            L_rg1, L_rg2, R_rg1, R_rg2,
            L_exc1, L_exc2, R_exc1, R_exc2,
            L_V0C_1, L_V0C_2,
            L_V1a_1, L_V1a_2,
            L_inh1, L_inh2, R_inh2,
            L_rc_1, L_rc_2,
            L_mnp1, L_mnp2, R_mnp1, R_mnp2,
            L_V0D, R_V0D,
            L_label,
            online_ramp_log,
            nn.online_ramp_weight,
            ramp_type="online_ramp_drive"
        )

        print("[INFO] Calling RIGHT utils")
        cpg_utils(
            nn, popfunc, conn,
            R_rg1, R_rg2, L_rg1, L_rg2,
            R_exc1, R_exc2, L_exc1, L_exc2,
            R_V0C_1, R_V0C_2,
            R_V1a_1, R_V1a_2,
            R_inh1, R_inh2, L_inh2,
            R_rc_1, R_rc_2,
            R_mnp1, R_mnp2, L_mnp1, L_mnp2,
            R_V0D, L_V0D,
            R_label,
            online_ramp_log,
            nn.online_ramp_weight,
            ramp_type="online_ramp_drive"
        )

    # ================= END ONLINE RAMP DRIVE EXPERIMENT ======================

    ##  ============================== ONLINE STEPWISE DRIVE EXPERIMENT (Bilateral Configured) ===============================
    if nn.online_stepwise_experiment and nn.stepwise_drive_experiment == 1:

        print("[INFO] Online Stepwise DRIVE Experiment is ON")
       

        # Step sizes / holds from config
        step_scales = list(nn.step_scales)      # e.g. [1, 2, 3]
        hold_ms     = float(nn.step_hold_ms)    # e.g. 500 ms
        step_ms = float(nn.time_resolution) * 10.0  # simulation chunk (consistent with your ramp/feedback loop)
        num_plateaus = len(step_scales)
        hold_steps = max(1, int(round(hold_ms / step_ms)))
        drive_param  = 'I_e'   # specified drive instrinsic parameter in NEST neurons 

        drive_baselines = []

        raw_targets = (
            [nn.online_stepwise_drive_targets]
            if isinstance(nn.online_stepwise_drive_targets, str)
            else nn.online_stepwise_drive_targets
        )

        base_map = {  # dictionary of baseline drives for different targets
            "L_V3_Flx.v3_bursting": I_e_tonic_v3_L_mean,
            "R_V3_Flx.v3_bursting": I_e_tonic_v3_R_mean,
            "L_V0D.v0d_bursting": I_e_tonic_inh_L_mean,
            "R_V0D.v0d_bursting": I_e_tonic_inh_R_mean,
        }

        drive_targets = popfunc.resolve_targets_to_nc(
            raw_targets,
            roots={
                "L_V0D": L_V0D,
                "R_V0D": R_V0D,
                "L_V3_Flx": L_V3_Flx,
                "R_V3_Flx": R_V3_Flx,
                "L_V3_Ext": L_V3_Ext,
                "R_V3_Ext": R_V3_Ext,
            }
        )

        x_base_map = {}

        for target_str, pop in zip(raw_targets, drive_targets):

            if target_str not in base_map:
                raise ValueError(f"No baseline defined for {target_str}")

            param_name = base_map[target_str]
            x_base_map[pop] = param_name

            print(f"[INFO] Baseline for {target_str} = {param_name:.4f} pA")

      
        baseline_hash = dict(x_base_map)

        # --- Logging ---
        online_stepwise_log = {
            "time": [],
            "mult": [],
            "drive": [],
            "plateaus": [],
        }

        # burn simulation
        init_time = 50.0
        nest.Simulate(init_time)
        t_start_wall = time.perf_counter()

        for k, mult in enumerate(step_scales):

            # Step sizes / holds from config
            mult = float(mult)

            # baseline dataset integrity 
            if x_base_map != baseline_hash:
                raise RuntimeError("Baseline values changed during experiment.")

            print(f"\n[STEP {k+1}/{num_plateaus}] mult={mult:.3f}")

            for pop, x_base in x_base_map.items():
                x_new = x_base * mult
                nest.SetStatus(pop, {drive_param: float(x_new)})

            t_step_start = float(nest.biological_time)

            # ---- Hold plateau ----
            for j in range(hold_steps):
                
                nest.Simulate(step_ms)
                t = float(nest.biological_time)

                # --- log time series for plotting ---
                online_stepwise_log["time"].append(t)
                online_stepwise_log["mult"].append(mult)
                online_stepwise_log["drive"].append(mult)  # multiplier sufficient

                # --- feedback L ---
                num_spikes_flx_L = popfunc.read_recent_spike_data(L_mnp1.spike_detector_motor)
                num_spikes_ext_L = popfunc.read_recent_spike_data(L_mnp2.spike_detector_motor)
                L_fb.send_muscle_activation(num_spikes_flx_L, num_spikes_ext_L)
                flx_1a_L, ext_1a_L, flx_1b_L, ext_1b_L, *_ = L_fb.receive_muscle_afferents()

                if nn.fb_rg_flx: nest.SetStatus(L_rg1.rg_flx_pg, {"rate": flx_1a_L})
                if nn.fb_rg_ext: nest.SetStatus(L_rg2.rg_ext_pg, {"rate": ext_1b_L})
                if nn.fb_v2b:    nest.SetStatus(L_inh1.v2b_pg, {"rate": flx_1a_L})
                if nn.fb_v1:     nest.SetStatus(L_inh2.v1_pg, {"rate": ext_1b_L})

                # --- feedback R ---
                num_spikes_flx_R = popfunc.read_recent_spike_data(R_mnp1.spike_detector_motor)
                num_spikes_ext_R = popfunc.read_recent_spike_data(R_mnp2.spike_detector_motor)
                R_fb.send_muscle_activation(num_spikes_flx_R, num_spikes_ext_R)
                flx_1a_R, ext_1a_R, flx_1b_R, ext_1b_R, *_ = R_fb.receive_muscle_afferents()

                if nn.fb_rg_flx: nest.SetStatus(R_rg1.rg_flx_pg, {"rate": flx_1a_R})
                if nn.fb_rg_ext: nest.SetStatus(R_rg2.rg_ext_pg, {"rate": ext_1b_R})
                if nn.fb_v2b:    nest.SetStatus(R_inh1.v2b_pg, {"rate": flx_1a_R})
                if nn.fb_v1:     nest.SetStatus(R_inh2.v1_pg, {"rate": ext_1b_R})

                print(
                    f"PLATEAU={k+1}/{num_plateaus} | "
                    f"hold={j+1:>3}/{hold_steps} | "
                    f"t={t:.1f} ms",
                    end="\r"
                )
                
            # plateau end time (after hold ms)
            t_step_end = float(nest.biological_time)

            if nn.low_locomotion_v0d_left and nn.low_locomotion_v0d_right:

                 # plateau summary for v0d
                L_v0d_out = popfunc.window_rates_from_spike_detector(
                    L_V0D.spike_detector, nn.v0d_pop_size_L, t_step_start, t_step_end
                )
                R_v0d_out = popfunc.window_rates_from_spike_detector(
                    R_V0D.spike_detector, nn.v0d_pop_size_L, t_step_start, t_step_end
                )

                print(f"| LEFT V0D spikes={L_v0d_out['n_spikes']} LEFT V0D mean_hz={L_v0d_out['mean_hz']:.3f}")
                print(f"| RIGHT V0D spikes={R_v0d_out['n_spikes']} LEFT V0D mean_hz={R_v0d_out['mean_hz']:.3f}")


            if nn.low_locomotion_v3_left and nn.low_locomotion_v3_right: 

                # plateau summary for v0d
                L_v3_out = popfunc.window_rates_from_spike_detector(
                    L_V3_Flx.spike_detector, nn.v3_pop_size, t_step_start, t_step_end
                )
                R_v3_out = popfunc.window_rates_from_spike_detector(
                    R_V3_Flx.spike_detector, nn.v3_pop_size, t_step_start, t_step_end
                )

                print(f"| LEFT V3 spikes={L_v0d_out['n_spikes']} LEFT V3 mean_hz={L_v0d_out['mean_hz']:.3f}")
                print(f"| RIGHT V3 spikes={R_v0d_out['n_spikes']} LEFT V3 mean_hz={R_v0d_out['mean_hz']:.3f}")


            # Store plateau boundaries + metrics
            online_stepwise_log["plateaus"].append({
                "k": k,
                "mult": float(mult),
                "drive": float(x_new),
                "t_start": float(t_step_start),
                "t_end": float(t_step_end),
            })

        t_stop_wall = time.perf_counter()
        print("\n" + "=" * 70)
        print(f"[INFO] Online Stepwise DRIVE completed in {t_stop_wall - t_start_wall:.2f} s")

        # ================= CPG UTILS (after stepwise) =================
        L_label = "LEFT"
        R_label = "RIGHT"

        print("[INFO] Calling LEFT utils")
        cpg_utils(
            nn, popfunc, conn,
            L_rg1, L_rg2, R_rg1, R_rg2,
            L_exc1, L_exc2, R_exc1, R_exc2,
            L_V0C_1, L_V0C_2,
            L_V1a_1, L_V1a_2,
            L_inh1, L_inh2, R_inh2,
            L_rc_1, L_rc_2,
            L_mnp1, L_mnp2, R_mnp1, R_mnp2,
            L_V0D, R_V0D,
            L_V3_Flx, R_V3_Flx,
            L_V3_Ext, R_V3_Ext,
            L_label,
            online_stepwise_log,
            drive_param,                     # replaces nn.online_stepwise_weight
            ramp_type="online_stepwise_drive"
        )

        print("[INFO] Calling RIGHT utils")
        cpg_utils(
            nn, popfunc, conn,
            R_rg1, R_rg2, L_rg1, L_rg2,
            R_exc1, R_exc2, L_exc1, L_exc2,
            R_V0C_1, R_V0C_2,
            R_V1a_1, R_V1a_2,
            R_inh1, R_inh2, L_inh2,
            R_rc_1, R_rc_2,
            R_mnp1, R_mnp2, L_mnp1, L_mnp2,
            R_V0D, L_V0D,
            R_V3_Flx, L_V3_Flx,
            R_V3_Ext, L_V3_Ext,
            R_label,
            online_stepwise_log,
            drive_param,
            ramp_type="online_stepwise_drive"
        )
    # =============== END ONLINE STEPWISE DRIVE EXPERIMENT ===============================


    # ================= ONLINE STEPWISE WEIGHT EXPERIMENT (Bilateral Configured) =================
    if nn.online_stepwise_experiment == 1 and nn.stepwise_weight_experiment == 1:

        # Amended so we can have a bi-weight stepwise ramp.  
        print("[INFO] Online Stepwise Weight Experiment is ON. ")
    
        # Step sizes / holds from config defining them all 
        step_scales = list(nn.step_scales)          # e.g. [1, 2, 3]
        hold_ms     = float(nn.step_hold_ms)        # e.g. 500 ms
        step_ms = float(nn.time_resolution) * 10.0  # simulation chunk (consistent with your ramp/feedback loop)
        num_plateaus = len(step_scales)
        hold_steps = max(1, int(round(hold_ms / step_ms)))
        
        bin_ms = 10.0
        connections_list = []
        connection_baselines = []

        # Log dict (keep keys compatible with plotting)
        online_stepwise_log = {
            "time": [],
            "mult": [],
            "mean_weight": [],
            "weight": [],
            "step_metrics": [],
            "plateaus": []
        }

        print(f"[INFO] step_ms = {step_ms:.2f} ms")
        print(f"[INFO] Stepwise scales = {step_scales} (n={num_plateaus})")

        # defining weights as list. 
        if not isinstance(nn.online_stepwise_weight, list):
            specified_weights = [nn.online_stepwise_weight]
        else:
            specified_weights = nn.online_stepwise_weight
     
        key_to_src_tgt = {
        "custom_v0d_rg_inh_L": (L_V0D.v0d_bursting, R_rg1.rg_exc_bursting), # source, target ... 
        "custom_v0d_rg_inh_R": (R_V0D.v0d_bursting, L_rg1.rg_exc_bursting),
        "custom_v3_rg_R": (R_V3_Flx.v3_bursting, L_rg1.rg_exc_bursting),
        "custom_v3_rg_L": (L_V3_Flx.v3_bursting, R_rg1.rg_exc_bursting),
        # etc...
        }

        # if instances for more than one specified weight 
        for key in specified_weights:

            if key not in key_to_src_tgt:
                raise ValueError(f"Unknown weight key {key}")

            src, tgt = key_to_src_tgt[key]
            
            # derving source and target 
            conn = nest.GetConnections(source=src, target=tgt)

            if len(conn) == 0:
                raise RuntimeError(f"No connections found for {key}")

            connections_list.append(conn)

            w_base = np.asarray(nest.GetStatus(conn, "weight"), dtype=float)
            connection_baselines.append(w_base.copy()) # this is acting as a baseline hash weight. safe keeping to ensure we are not overwriting this. 
            print(f"[INFO] Synapse {key} | Mean={w_base.mean():.3f} | "f"Min={w_base.min():.3f} | Max={w_base.max():.3f}")

        
        init_time = 50.0
        nest.Simulate(init_time)
        t_start = time.perf_counter()

        for k, mult in enumerate(step_scales):
            
            mult = float(mult)

            print(f"\n[STEP {k+1}/{len(step_scales)}] mult={mult:.3f}")

             # Apply weight scaling per weight 
            for conn, w_base in zip(connections_list, connection_baselines):

                n_samp = min(200, len(conn))
                idx = np.random.choice(len(conn), size=n_samp, replace=False)
                w_pre_all = np.asarray(nest.GetStatus(conn, "weight"), dtype=float)
                w_pre = w_pre_all[idx]
            

                w_new = w_base * mult
                
                # applying new weight. 
                nest.SetStatus(conn, [{"weight": float(w)} for w in w_new])

                mw = float(np.mean(w_new))
                #print(f"[INFO] Applying mean_w={mw:.3f} to connection")

                w_post_all = np.asarray(nest.GetStatus(conn, "weight"), dtype=float)
                w_post = w_post_all[idx]

            # plateau start time (after applying weights)
            t_step_start = float(nest.biological_time)

            # ---- Hold plateau ----
            for j in range(hold_steps):
                
                nest.Simulate(step_ms)
                
                t = float(nest.biological_time)

                online_stepwise_log["time"].append(t)
                online_stepwise_log["mult"].append(mult)
                online_stepwise_log["mean_weight"].append(mw)
                online_stepwise_log["weight"].append(mw)

                # --- feedback L ---
                num_spikes_flx_L = popfunc.read_recent_spike_data(L_mnp1.spike_detector_motor)
                num_spikes_ext_L = popfunc.read_recent_spike_data(L_mnp2.spike_detector_motor)
                L_fb.send_muscle_activation(num_spikes_flx_L, num_spikes_ext_L)
                flx_1a_L, ext_1a_L, flx_1b_L, ext_1b_L, *_ = L_fb.receive_muscle_afferents()

                if nn.fb_rg_flx: nest.SetStatus(L_rg1.rg_flx_pg, {"rate": flx_1a_L})
                if nn.fb_rg_ext: nest.SetStatus(L_rg2.rg_ext_pg, {"rate": ext_1b_L})
                if nn.fb_v2b:    nest.SetStatus(L_inh1.v2b_pg, {"rate": flx_1a_L})
                if nn.fb_v1:     nest.SetStatus(L_inh2.v1_pg, {"rate": ext_1b_L})

                # --- feedback R ---
                num_spikes_flx_R = popfunc.read_recent_spike_data(R_mnp1.spike_detector_motor)
                num_spikes_ext_R = popfunc.read_recent_spike_data(R_mnp2.spike_detector_motor)
                R_fb.send_muscle_activation(num_spikes_flx_R, num_spikes_ext_R)
                flx_1a_R, ext_1a_R, flx_1b_R, ext_1b_R, *_ = R_fb.receive_muscle_afferents()

                if nn.fb_rg_flx: nest.SetStatus(R_rg1.rg_flx_pg, {"rate": flx_1a_R})
                if nn.fb_rg_ext: nest.SetStatus(R_rg2.rg_ext_pg, {"rate": ext_1b_R})
                if nn.fb_v2b:    nest.SetStatus(R_inh1.v2b_pg, {"rate": flx_1a_R})
                if nn.fb_v1:     nest.SetStatus(R_inh2.v1_pg, {"rate": ext_1b_R})

                print(
                    f"STEP={k+1}/{num_plateaus} mult={mult:.2f} | "
                    f"hold={j+1:>3}/{hold_steps} | t={t:.1f} ms | mean w={mw:.3f}",
                    end="\r"
                )

            # plateau end time
            t_step_end = float(nest.biological_time)

            print(
                f"\n[PLATEAU {k+1} STATS /{num_plateaus}] mult={mult:.2f} "
                f"| t={t_step_start:.1f}->{t_step_end:.1f} ms "
            )

            # plateau end time (after hold ms)
            t_step_end = float(nest.biological_time)

            if nn.low_locomotion_v0d_left and nn.low_locomotion_v0d_right:

                 # plateau summary for v0d
                L_v0d_out = popfunc.window_rates_from_spike_detector(
                    L_V0D.spike_detector, nn.v0d_pop_size_L, t_step_start, t_step_end
                )
                R_v0d_out = popfunc.window_rates_from_spike_detector(
                    R_V0D.spike_detector, nn.v0d_pop_size_L, t_step_start, t_step_end
                )

                print(f"| LEFT V0D spikes={L_v0d_out['n_spikes']} LEFT V0D mean_hz={L_v0d_out['mean_hz']:.3f}")
                print(f"| RIGHT V0D spikes={R_v0d_out['n_spikes']} LEFT V0D mean_hz={R_v0d_out['mean_hz']:.3f}")


            if nn.low_locomotion_v3_left and nn.low_locomotion_v3_right: 

                # plateau summary for v0d
                L_v3_out = popfunc.window_rates_from_spike_detector(
                    L_V3_Flx.spike_detector, nn.v3_pop_size, t_step_start, t_step_end
                )
                R_v3_out = popfunc.window_rates_from_spike_detector(
                    R_V3_Flx.spike_detector, nn.v3_pop_size, t_step_start, t_step_end
                )

                print(f"| LEFT V3 spikes={L_v3_out['n_spikes']} LEFT V3 mean_hz={L_v3_out['mean_hz']:.3f}")
                print(f"| RIGHT V3 spikes={R_v3_out['n_spikes']} LEFT V3 mean_hz={R_v3_out['mean_hz']:.3f}")


            # phase analysis 
            mnp1 = popfunc.window_rates_from_spike_detector(
            L_mnp1.spike_detector_motor,
            nn.num_motor_neurons,
            t_step_start,
            t_step_end,
            return_spike_times=True
            )

            #print(f"[DEBUG] MNP1 Spike Times MS: {mnp1['spike_times_ms']}")

            mnp2 = popfunc.window_rates_from_spike_detector(
                R_mnp1.spike_detector_motor,
                nn.num_motor_neurons,
                t_step_start,
                t_step_end,
                return_spike_times=True
            )

            #print(f"[DEBUG] MNP2 Spike Times MS: {mnp2['spike_times_ms']}")

            p1 = popfunc.minmax_safe(
                popfunc.binned_rate_from_spike_times(
                    mnp1["spike_times_ms"], t_step_start, t_step_end, bin_ms
                )
            )

            p2 = popfunc.minmax_safe(
                popfunc.binned_rate_from_spike_times(
                    mnp2["spike_times_ms"], t_step_start, t_step_end, bin_ms
                )
            )

            if p1 is None or p2 is None:
                phase_mean = np.nan
                phase_var = np.nan
                phase_cv = np.nan
                f1 = np.nan
                f2 = np.nan
                diag = {"reason": "flat_signal"}

            else:
                phase_mean, phase_var, phase_cv, f1, f2, diag = popfunc.calculate_peak_to_peak_phase_diag(
                    p1, p2,
                    bin_ms=bin_ms,
                    min_peak_height=0.3,
                    min_dist_ms=300.0,
                    prominence=0.05
                )
            
            print(f"| MNP1 Ipsi and Contra Phase: {phase_mean}, Variance: {phase_var}, CV: {phase_cv}")
            #print(f"[DEBUG] MNP1 Ipsi and Contra F1: {f1}, F2: {f2}, diag: {diag}")


            online_stepwise_log["plateaus"].append({
                "k": int(k),
                "mult": float(mult),
                "t_start": float(t_step_start),
                "t_end": float(t_step_end),
                "phase_deg_mean": float(phase_mean),
                "phase_deg_var": float(phase_var),
                "freq_L_hz": float(f1),
                "freq_R_hz": float(f2),
            })




        t_stop = time.perf_counter()
        print(f"[INFO] Online Stepwise completed in {t_stop - t_start:.2f} s")

        # ================= CPG UTILS (after stepwise) =================
        L_label = "LEFT"
        R_label = "RIGHT"

        print("[INFO] Calling LEFT utils")
        cpg_utils(
            nn, popfunc, conn,
            L_rg1, L_rg2, R_rg1, R_rg2,
            L_exc1, L_exc2, R_exc1, R_exc2,
            L_V0C_1, L_V0C_2,
            L_V1a_1, L_V1a_2,
            L_inh1, L_inh2, R_inh2,
            L_rc_1, L_rc_2,
            L_mnp1, L_mnp2, R_mnp1, R_mnp2,
            L_V0D, R_V0D,
            L_V3_Flx, R_V3_Flx,
            L_V3_Ext, R_V3_Ext,
            L_label,
            online_stepwise_log,
            nn.online_stepwise_weight,
            ramp_type="online_stepwise_weight"
        )

        print("[INFO] Calling RIGHT utils")
        cpg_utils(
            nn, popfunc, conn,
            R_rg1, R_rg2, L_rg1, L_rg2,
            R_exc1, R_exc2, L_exc1, L_exc2,
            R_V0C_1, R_V0C_2,
            R_V1a_1, R_V1a_2,
            R_inh1, R_inh2, L_inh2,
            R_rc_1, R_rc_2,
            R_mnp1, R_mnp2, L_mnp1, L_mnp2,
            R_V0D, L_V0D,
            R_V3_Flx, L_V3_Flx,
            R_V3_Ext, L_V3_Ext, 
            R_label,
            online_stepwise_log,
            nn.online_stepwise_weight,
            ramp_type="online_stepwise_weight"
        )


    # if OFFLINE AND ONLINE experiments are not activated from the config file run simulations normally. STATIC WEIGHTS. 
    if (nn.online_ramp_experiment == 0) and (nn.online_stepwise_experiment == 0):

        print("[INFO] OFFLINE, ONLINE Experiments have been turned OFF - Running Normal Static Simulation. ")

        # ================= SIMULATION FOR BOTH L_ AND R_ =================

        init_time = 50
        nest.Simulate(init_time)
        num_steps = int(nn.sim_time / nn.time_resolution)
        t_start = time.perf_counter()

        # this loops every no.steps for each time_resolution 
        for i in range(int(num_steps / 10) - init_time):
            nest.Simulate(nn.time_resolution * 10)

            # --- L_ side feedback ---
            num_spikes_flx_L = popfunc.read_recent_spike_data(L_mnp1.spike_detector_motor)
            num_spikes_ext_L = popfunc.read_recent_spike_data(L_mnp2.spike_detector_motor)
            L_fb.send_muscle_activation(num_spikes_flx_L, num_spikes_ext_L)
            flx_1a_feedback_L, ext_1a_feedback_L, flx_1b_feedback_L, ext_1b_feedback_L, flx_11_feedback_L, ext_11_feedback_L = L_fb.receive_muscle_afferents()
            
            if nn.fb_rg_flx == 1:
                nest.SetStatus(L_rg1.rg_flx_pg, {"rate": flx_1a_feedback_L})
            if nn.fb_rg_ext == 1:
                nest.SetStatus(L_rg2.rg_ext_pg, {"rate": ext_1b_feedback_L})
            if nn.fb_v2b == 1:
                nest.SetStatus(L_inh1.v2b_pg, {"rate": flx_1a_feedback_L})
            if nn.fb_v1 == 1:
                nest.SetStatus(L_inh2.v1_pg, {"rate": ext_1b_feedback_L})
            if nn.fb_1a_flx == 1:
                nest.SetStatus(L_V1a_1.v1a_1_pg, {"rate": ext_1b_feedback_L})
            if nn.fb_1a_ext == 1:
                nest.SetStatus(L_V1a_2.v1a_2_pg, {"rate": flx_1a_feedback_L})

            # --- R_ side feedback ---
            num_spikes_flx_R = popfunc.read_recent_spike_data(R_mnp1.spike_detector_motor)
            num_spikes_ext_R = popfunc.read_recent_spike_data(R_mnp2.spike_detector_motor)
            R_fb.send_muscle_activation(num_spikes_flx_R, num_spikes_ext_R)
            flx_1a_feedback_R, ext_1a_feedback_R, flx_1b_feedback_R, ext_1b_feedback_R, flx_11_feedback_R, ext_11_feedback_R = R_fb.receive_muscle_afferents()
            
            if nn.fb_rg_flx == 1:
                nest.SetStatus(R_rg1.rg_flx_pg, {"rate": flx_1a_feedback_R})
            if nn.fb_rg_ext == 1:
                nest.SetStatus(R_rg2.rg_ext_pg, {"rate": ext_1b_feedback_R})
            if nn.fb_v2b == 1:
                nest.SetStatus(R_inh1.v2b_pg, {"rate": flx_1a_feedback_R})
            if nn.fb_v1 == 1:
                nest.SetStatus(R_inh2.v1_pg, {"rate": ext_1b_feedback_R})
            if nn.fb_1a_flx == 1:
                nest.SetStatus(R_V1a_1.v1a_1_pg, {"rate": ext_1b_feedback_R})
            if nn.fb_1a_ext == 1:
                nest.SetStatus(R_V1a_2.v1a_2_pg, {"rate": flx_1a_feedback_R})

            print("t = " + str(nest.biological_time), end="\r")

        t_stop = time.perf_counter()
        print('Simulation completed. It took ', round(t_stop - t_start, 2), ' seconds.')

        # ================= CPG UTILS FOR BOTH SIDES =================
        L_label = "LEFT"
        R_label = "RIGHT"

        print("[INFO] Calling LEFT utilis")

        cpg_utils(
            nn, popfunc, conn,
            L_rg1, L_rg2, R_rg1, R_rg2, 
            L_exc1, L_exc2, R_exc1, R_exc2, 
            L_V0C_1, L_V0C_2,
            L_V1a_1, L_V1a_2,
            L_inh1, # left v2b
            L_inh2, # left v1
            R_inh2, # right (contra) v1 
            L_rc_1, L_rc_2,
            L_mnp1, L_mnp2, R_mnp1, R_mnp2,
            L_V0D, R_V0D,
            L_V3_Flx, R_V3_Flx,
            L_V3_Ext, R_V3_Ext, 
            L_label
        )

        print("[INFO] Calling RIGHT utilis")

        cpg_utils(
            nn, popfunc, conn,
            R_rg1, R_rg2, L_rg1, L_rg2,
            R_exc1, R_exc2, L_exc1, L_exc2,
            R_V0C_1, R_V0C_2,
            R_V1a_1, R_V1a_2,
            R_inh1, # Right V2b
            R_inh2, #  Right V1
            L_inh2, # Left (contra) V1
            R_rc_1, R_rc_2,
            R_mnp1, R_mnp2, L_mnp1, L_mnp2,
            R_V0D, L_V0D,
            R_V3_Flx, L_V3_Flx,
            R_V3_Ext, L_V3_Ext,
            R_label
        )