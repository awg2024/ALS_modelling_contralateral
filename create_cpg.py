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
import create_commissural_pop as commissural

import create_mnp as mnp
import calculate_stability_metrics as calc
import send_receive_feedback as interface_fb
from cpg_data_utils import cpg_utils

warnings.filterwarnings('ignore')

#Create neuron populations - NEST

# =====================================
# LEFT POPULATION 
L_rg1 = flx_rg.create_rg_population()
L_rg2 = ext_rg.create_rg_population()


# exc1 = flx v2a tonic 
# exc2 = ext v2a tonic 
L_exc1 = exc.create_exc_inter_population()
L_exc2 = exc.create_exc_inter_population()

# exc1 = flx v2a burst 
# exc2 = ext v2a burst 
L_exc1_burst = inter.interneuron_population()


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
L_V0V = commissural.commissural_population()
# =====================================


# =====================================
# RIGHTPOPULATION 
R_rg1 = flx_rg.create_rg_population()
R_rg2 = ext_rg.create_rg_population()

# exc1 = flx v2a tonic 
# exc2 = ext v2a tonic 
R_exc1 = exc.create_exc_inter_population()
R_exc2 = exc.create_exc_inter_population()

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
R_V0V = commissural.commissural_population()
# =====================================


# CONFIGURING LEFT CONNECTIONS CPG. 
if nn.remove_descending_drive==0:

    # V0c_1 = FLx, V0c_2 = Ext 
    L_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
    L_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
    
    L_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback')
    L_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback')

elif nn.remove_descending_drive==1:

    L_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='none')
    L_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='none')
    
    L_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='none')
    L_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='none')


if nn.slow_syn_bias == 'flx':
    print('Slow synaptic dynamics applied to Flexor side only.')
    L_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    L_rc_2.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    L_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled')
    L_mnp2.create_mnp(pop_type='mnp_slow_syn_disabled')

elif nn.slow_syn_bias == 'ext':
    
    print('Slow synaptic dynamics applied to Extensor side only.')
    L_rc_1.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    L_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    L_mnp1.create_mnp(pop_type='mnp_slow_syn_disabled')
    L_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled')

else:
    # default setting slow synaptic dynamics... slow of locomotion...

    print('Slow synaptic dynamics applied to Flexor and Extensor.')
    L_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    L_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    L_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled')
    L_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled')


# flexor v2a 
L_exc1_burst.create_interneuron_population(
    pop_type='v2a',
    self_connection='none',
    firing_behavior='bursting',
    pop_size=nn.v2a_burst_pop_size,
    input_type='none'
)

#Connect rg neurons to V2a excitatory interneuron populations
# FOR EXC... 
# 1 - FLX
# 2 - EXT
conn.create_connections(L_rg1.rg_exc_bursting, L_exc1.exc_inter_tonic,'custom_rg_v2a')
conn.create_connections(L_rg1.rg_exc_tonic, L_exc1.exc_inter_tonic,'custom_rg_v2a')

conn.create_connections(L_rg2.rg_exc_bursting, L_exc2.exc_inter_tonic,'custom_rg_v2a')
conn.create_connections(L_rg2.rg_exc_tonic, L_exc2.exc_inter_tonic,'custom_rg_v2a')

#Connect V2a excitatory interneuron populations to motor neurons
conn.create_connections(L_exc1.exc_inter_tonic, L_mnp1.motor_neuron_pop,'custom_v2a_mn')
conn.create_connections(L_exc2.exc_inter_tonic, L_mnp2.motor_neuron_pop,'custom_v2a_mn')

#Connect rg neurons to V1a excitatory interneuron populations
conn.create_connections(L_rg1.rg_exc_bursting, L_V1a_2.interneuron_pop,'custom_rg_v1a')
conn.create_connections(L_rg1.rg_exc_tonic, L_V1a_2.interneuron_pop,'custom_rg_v1a')

conn.create_connections(L_rg2.rg_exc_bursting, L_V1a_1.interneuron_pop,'custom_rg_v1a')
conn.create_connections(L_rg2.rg_exc_tonic, L_V1a_1.interneuron_pop,'custom_rg_v1a')

#Connect rg neurons to V0c interneurons
conn.create_connections(L_rg1.rg_exc_bursting, L_V0C_1.v0c_tonic,'custom_rg_v0c')
conn.create_connections(L_rg1.rg_exc_tonic, L_V0C_1.v0c_tonic,'custom_rg_v0c')

conn.create_connections(L_rg2.rg_exc_bursting, L_V0C_2.v0c_tonic,'custom_rg_v0c')
conn.create_connections(L_rg2.rg_exc_tonic, L_V0C_2.v0c_tonic,'custom_rg_v0c')

#Connect V0c to motor neurons
conn.create_connections(L_V0C_1.v0c_tonic, L_mnp1.motor_neuron_pop,'custom_v0c_mn')
conn.create_connections(L_V0C_2.v0c_tonic, L_mnp2.motor_neuron_pop,'custom_v0c_mn')

#Connect V1a interneurons to contralateral V1a interneurons
conn.create_connections(L_V1a_2.interneuron_pop, L_V1a_1.interneuron_pop,'custom_v1a_v1a')
conn.create_connections(L_V1a_1.interneuron_pop, L_V1a_2.interneuron_pop,'custom_v1a_v1a')

#Connect V1a to motor neurons
conn.create_connections(L_V1a_1.interneuron_pop, L_mnp1.motor_neuron_pop,'custom_v1a_mn')
conn.create_connections(L_V1a_2.interneuron_pop, L_mnp2.motor_neuron_pop,'custom_v1a_mn')

#Connect RC interneurons to V1a interneurons
conn.create_connections(L_rc_1.interneuron_pop, L_V1a_2.interneuron_pop,'custom_rc_v1a')
conn.create_connections(L_rc_2.interneuron_pop, L_V1a_1.interneuron_pop,'custom_rc_v1a')

#Connect RC interneurons to contralateral RC interneurons
conn.create_connections(L_rc_1.interneuron_pop, L_rc_2.interneuron_pop,'custom_rc_rc')
conn.create_connections(L_rc_2.interneuron_pop, L_rc_1.interneuron_pop,'custom_rc_rc')

#Connect RC interneurons to motor neurons
conn.create_connections(L_rc_1.interneuron_pop, L_mnp1.motor_neuron_pop,'custom_rc_mn')
conn.create_connections(L_rc_2.interneuron_pop, L_mnp2.motor_neuron_pop,'custom_rc_mn')
conn.create_connections(L_mnp1.motor_neuron_pop, L_rc_1.interneuron_pop,'custom_mn_rc')
conn.create_connections(L_mnp2.motor_neuron_pop, L_rc_2.interneuron_pop,'custom_mn_rc')

if nn.rgs_connected == 1:
    L_inh1 = inh.create_inh_inter_population('V2b')  # V2b
    
    L_inh2 = inh.create_inh_inter_population('V1')  # V1

    # Connect excitatory rg neurons to V1/V2b inhibitory populations
    conn.create_connections(L_rg1.rg_exc_bursting, L_inh1.inh_inter_tonic, 'custom_rg_v2b')
    #conn.create_connections(L_rg1.rg_exc_tonic, L_inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(L_rg2.rg_exc_bursting, L_inh2.inh_inter_tonic, 'custom_rg_v1')
    #conn.create_connections(L_rg2.rg_exc_tonic, L_inh2.inh_inter_tonic, 'custom_rg_v1')

    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_exc_bursting,'custom_v2b_rg')
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_exc_tonic,'custom_v2b_rg')
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_inh_bursting,'custom_v2b_rg')
    conn.create_connections(L_inh1.inh_inter_tonic, L_rg2.rg_inh_tonic,'custom_v2b_rg')

    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_exc_bursting,'custom_v1_rg')
    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_exc_tonic,'custom_v1_rg')
    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_inh_bursting,'custom_v1_rg')
    conn.create_connections(L_inh2.inh_inter_tonic, L_rg1.rg_inh_tonic,'custom_v1_rg')

    #Connect V1/V2b inhibitory populations to V2a
    conn.create_connections(L_inh1.inh_inter_tonic, L_exc2.exc_inter_tonic,'custom_v2b_v2a')
    conn.create_connections(L_inh2.inh_inter_tonic, L_exc1.exc_inter_tonic,'custom_v1_v2a')

    if nn.v1v2b_mn_connected==1:
        #Connect V1/V2b inhibitory populations to motor neurons
        conn.create_connections(L_inh1.inh_inter_tonic, L_mnp2.motor_neuron_pop,'custom_v2b_mn')
        conn.create_connections(L_inh2.inh_inter_tonic, L_mnp1.motor_neuron_pop,'custom_v1_mn')
   
    # connect excitatory rg neurons
    conn.create_connections(L_rg1.rg_exc_bursting, L_rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(L_rg1.rg_exc_bursting, L_rg2.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(L_rg1.rg_exc_tonic, L_rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(L_rg1.rg_exc_tonic, L_rg2.rg_exc_tonic,'custom_rg_rg')

    conn.create_connections(L_rg2.rg_exc_bursting, L_rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(L_rg2.rg_exc_bursting, L_rg1.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(L_rg2.rg_exc_tonic, L_rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(L_rg2.rg_exc_tonic, L_rg1.rg_exc_tonic,'custom_rg_rg')

    # =======================================================
    # V0D CONTRALATERAL CODE BELOW 
    # V0d bursting at low locomotion and tonic at high​ locomotion 

    # create contralateral interneuron populations - BURSTING AND TONIC SUBTYPES 
    L_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='tonic', pop_size=nn.v0d_pop_size, input_type='none') 
    L_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='bursting', pop_size=nn.v0d_pop_size, input_type='none') 

    if nn.low_locomotion_v0d_left == 1: 
        
        print("Creating Connections for LEFT V0d - LOW LOCOMOTION")

        # RG_F → V0d_bursting - custom_rg_v0d generic connection we are applying across all connections 
        conn.create_connections(L_rg1.rg_exc_bursting, L_V0D.v0d_bursting, 'custom_rg_v0d_L')

        # V0d_bursting → RG_F CONTRA (inhibitory, correct) - this is the we are wanting to manipulate.
        conn.create_connections(L_V0D.v0d_bursting, R_rg1.rg_exc_bursting, 'custom_v0d_rg_inh_L')

    if nn.high_locomotion_v0d_left == 1: 
        
        print("Creating Connections for LEFT V0d - HIGH LOCOMOTION")

        # tonic connections v0d ipsilateral from both rg bursting and exc
        conn.create_connections(L_rg1.rg_exc_tonic, L_V0D.v0d_tonic, 'custom_rg_v0d_L')

        conn.create_connections(L_rg2.rg_inh_tonic, L_V0D.v0d_tonic, 'custom_rg_v0d_inh_L')

         # tonic v0d to contralateral rg tonic 
        conn.create_connections(L_V0D.v0d_tonic, R_rg1.rg_exc_tonic, 'custom_v0d_rg_inh_L')

    # =======================================================
    # V0V CONTRALATERAL CODE BELOW
    # V0v bursting at high locomotion and tonic at low locomotion 
    
    L_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='descending') 
    L_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='bursting', pop_size=nn.v0v_pop_size, input_type='descending') 
    
    if nn.low_locomotion_v0v_left == 1: 

        print("Creating connections for LEFT V0v - LOW LOCOMOTION ")
        
        # We are using seperate weights compared to the original architecture for pre-existing connections
        # this allows the seperate of different connections for v0v associated circuity. 

        # CONNECTION UNO: L_rg1 tonic input to L_V2a tonic 
        conn.create_connections(L_rg1.rg_exc_tonic, L_exc1.exc_inter_tonic, 'custom_rg_v2a_v0vconn')
        
        # tonic V0V receives tonic V2a (low frequency mode)
        conn.create_connections(L_exc1.exc_inter_tonic, L_V0V.v0v_tonic, 'custom_v2a_v0v')

        # tonic V0V excites V1 
        conn.create_connections(L_V0V.v0v_tonic, L_inh2.inh_inter_tonic, 'custom_v0v_v1')

        # V1 inhibits the RG_flx 
        conn.create_connections(L_inh2.inh_inter_tonic, R_rg1.rg_exc_tonic, 'custom_v1_rg_v0vconn')

    if nn.high_locomotion_v0v_left == 1: 

        print("Creating connections for LEFT V0v - HIGH LOCOMOTION")

        # L_rg1 bursting input to L_V2a bursting. 
        conn.create_connections(L_rg1.rg_exc_bursting, L_exc1.exc_inter_burst, 'custom_rg_v2a')
        
        # bursting V0V receives tonic V2a (low frequency mode)
        conn.create_connections(L_exc1.exc_inter_burst, L_V0V.v0v_bursting, 'custom_v2a_v0v')

        # tonic V0V excites V1 
        conn.create_connections(L_V0V.v0v_bursting, L_inh2.inh_inter_bursting, 'custom_v0v_v1')

        # V1 inhibits the RG_flx 
        conn.create(L_inh2.inh_inter_bursting, R_rg1.rg_exc_bursting,  'custom_v1_rg')
        

  
  # ================================ END OF LEFT SIDE CPG CODE ==========================================================


if nn.remove_descending_drive==0:
    
    R_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
    R_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='descending')
    
    R_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback')
    R_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='sensory_feedback')

elif nn.remove_descending_drive==1:
    
    R_V0C_1.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='none')
    R_V0C_2.create_commissural_population(pop_type='V0C',self_connection='none',firing_behavior='tonic',pop_size=nn.v0c_pop_size,input_type='none')
   
    R_V1a_1.create_interneuron_population(pop_type='V1a_1',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='none')
    R_V1a_2.create_interneuron_population(pop_type='V1a_2',self_connection='none',firing_behavior='tonic',pop_size=nn.v1a_pop_size,input_type='none')

if nn.slow_syn_bias == 'flx':
    print('Slow synaptic dynamics applied to Flexor side only.')
    R_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    R_rc_2.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    R_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled')
    R_mnp2.create_mnp(pop_type='mnp_slow_syn_disabled')
elif nn.slow_syn_bias == 'ext':
    print('Slow synaptic dynamics applied to Extensor side only.')
    R_rc_1.create_interneuron_population(pop_type='rc_slow_syn_disabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    R_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    R_mnp1.create_mnp(pop_type='mnp_slow_syn_disabled')
    R_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled')
else:
    print('Slow synaptic dynamics applied to Flexor and Extensor.')
    R_rc_1.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    R_rc_2.create_interneuron_population(pop_type='rc_slow_syn_enabled',self_connection='none',firing_behavior='bursting',pop_size=nn.rc_pop_size,input_type='none')
    R_mnp1.create_mnp(pop_type='mnp_slow_syn_enabled')
    R_mnp2.create_mnp(pop_type='mnp_slow_syn_enabled')


   
    # V0c populations have already been defined

    if nn.contralateral_projections_v0c_right == 1: 

        # 1 = Flx, 2 = Ext.
        # flexing V0c goes to flx Right MNP - exciting it 
        # extensing V0c goes to the ext Right MNP - exciting it  

        conn.create_connections(L_V0C_1.v0c_tonic, R_mnp1.motor_neuron_pop, 'custom_v0c_mnp_flx')
        conn.create_connections(L_V0C_2.v0c_tonic, R_mnp2.motor_neuron_pop, 'custom_v0c_mnp_ext')

# flexor v2a 
R_exc1_burst.create_interneuron_population(
    pop_type='v2a',
    self_connection='none',
    firing_behavior='bursting',
    pop_size=nn.v2a_burst_pop_size,
    input_type='none'
)

#Connect rg neurons to V2a excitatory interneuron populations
conn.create_connections(R_rg1.rg_exc_bursting, R_exc1.exc_inter_tonic,'custom_rg_v2a')
conn.create_connections(R_rg1.rg_exc_tonic, R_exc1.exc_inter_tonic,'custom_rg_v2a')

conn.create_connections(R_rg2.rg_exc_bursting, R_exc2.exc_inter_tonic,'custom_rg_v2a')
conn.create_connections(R_rg2.rg_exc_tonic, R_exc2.exc_inter_tonic,'custom_rg_v2a')

#Connect V2a excitatory interneuron populations to motor neurons
conn.create_connections(R_exc1.exc_inter_tonic, R_mnp1.motor_neuron_pop,'custom_v2a_mn')
conn.create_connections(R_exc2.exc_inter_tonic, R_mnp2.motor_neuron_pop,'custom_v2a_mn')

#Connect rg neurons to V1a excitatory interneuron populations
conn.create_connections(R_rg1.rg_exc_bursting, R_V1a_2.interneuron_pop,'custom_rg_v1a')
conn.create_connections(R_rg1.rg_exc_tonic, R_V1a_2.interneuron_pop,'custom_rg_v1a')

conn.create_connections(R_rg2.rg_exc_bursting, R_V1a_1.interneuron_pop,'custom_rg_v1a')
conn.create_connections(R_rg2.rg_exc_tonic, R_V1a_1.interneuron_pop,'custom_rg_v1a')

#Connect rg neurons to V0c interneurons
conn.create_connections(R_rg1.rg_exc_bursting, R_V0C_1.v0c_tonic,'custom_rg_v0c')
conn.create_connections(R_rg1.rg_exc_tonic, R_V0C_1.v0c_tonic,'custom_rg_v0c')

conn.create_connections(R_rg2.rg_exc_bursting, R_V0C_2.v0c_tonic,'custom_rg_v0c')
conn.create_connections(R_rg2.rg_exc_tonic, R_V0C_2.v0c_tonic,'custom_rg_v0c')

#Connect V0c to motor neurons
conn.create_connections(R_V0C_1.v0c_tonic, R_mnp1.motor_neuron_pop,'custom_v0c_mn')
conn.create_connections(R_V0C_2.v0c_tonic, R_mnp2.motor_neuron_pop,'custom_v0c_mn')

#Connect V1a interneurons to contralateral V1a interneurons
conn.create_connections(R_V1a_2.interneuron_pop, R_V1a_1.interneuron_pop,'custom_v1a_v1a')
conn.create_connections(R_V1a_1.interneuron_pop, R_V1a_2.interneuron_pop,'custom_v1a_v1a')

#Connect V1a to motor neurons
conn.create_connections(R_V1a_1.interneuron_pop, R_mnp1.motor_neuron_pop,'custom_v1a_mn')
conn.create_connections(R_V1a_2.interneuron_pop, R_mnp2.motor_neuron_pop,'custom_v1a_mn')

#Connect RC interneurons to V1a interneurons
conn.create_connections(R_rc_1.interneuron_pop, R_V1a_2.interneuron_pop,'custom_rc_v1a')
conn.create_connections(R_rc_2.interneuron_pop, R_V1a_1.interneuron_pop,'custom_rc_v1a')

#Connect RC interneurons to contralateral RC interneurons
conn.create_connections(R_rc_1.interneuron_pop, R_rc_2.interneuron_pop,'custom_rc_rc')
conn.create_connections(R_rc_2.interneuron_pop, R_rc_1.interneuron_pop,'custom_rc_rc')

#Connect RC interneurons to motor neurons
conn.create_connections(R_rc_1.interneuron_pop, R_mnp1.motor_neuron_pop,'custom_rc_mn')
conn.create_connections(R_rc_2.interneuron_pop, R_mnp2.motor_neuron_pop,'custom_rc_mn')
conn.create_connections(R_mnp1.motor_neuron_pop, R_rc_1.interneuron_pop,'custom_mn_rc')
conn.create_connections(R_mnp2.motor_neuron_pop, R_rc_2.interneuron_pop,'custom_mn_rc')

if nn.rgs_connected == 1:
    R_inh1 = inh.create_inh_inter_population('V2b')  # V2b
    R_inh2 = inh.create_inh_inter_population('V1')  # V1

    # Connect excitatory rg neurons to V1/V2b inhibitory populations
    conn.create_connections(R_rg1.rg_exc_bursting, R_inh1.inh_inter_tonic, 'custom_rg_v2b')
    #conn.create_connections(R_rg1.rg_exc_tonic, R_inh1.inh_inter_tonic, 'custom_rg_v2b')
    conn.create_connections(R_rg2.rg_exc_bursting, R_inh2.inh_inter_tonic, 'custom_rg_v1')
    #conn.create_connections(R_rg2.rg_exc_tonic, R_inh2.inh_inter_tonic, 'custom_rg_v1')

    #Connect V1/V2b inhibitory populations to all rg neurons
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_exc_bursting,'custom_v2b_rg')
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_exc_tonic,'custom_v2b_rg')
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_inh_bursting,'custom_v2b_rg')
    conn.create_connections(R_inh1.inh_inter_tonic, R_rg2.rg_inh_tonic,'custom_v2b_rg')

    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_exc_bursting,'custom_v1_rg')
    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_exc_tonic,'custom_v1_rg')
    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_inh_bursting,'custom_v1_rg')
    conn.create_connections(R_inh2.inh_inter_tonic, R_rg1.rg_inh_tonic,'custom_v1_rg')

    #Connect V1/V2b inhibitory populations to V2a
    conn.create_connections(R_inh1.inh_inter_tonic, R_exc2.exc_inter_tonic,'custom_v2b_v2a')
    conn.create_connections(R_inh2.inh_inter_tonic, R_exc1.exc_inter_tonic,'custom_v1_v2a')

    if nn.v1v2b_mn_connected==1:
        #Connect V1/V2b inhibitory populations to motor neurons
        conn.create_connections(R_inh1.inh_inter_tonic, R_mnp2.motor_neuron_pop,'custom_v2b_mn')
        conn.create_connections(R_inh2.inh_inter_tonic, R_mnp1.motor_neuron_pop,'custom_v1_mn')

    #Connect excitatory rg neurons
    conn.create_connections(R_rg1.rg_exc_bursting, R_rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(R_rg1.rg_exc_bursting, R_rg2.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(R_rg1.rg_exc_tonic, R_rg2.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(R_rg1.rg_exc_tonic, R_rg2.rg_exc_tonic,'custom_rg_rg')

    conn.create_connections(R_rg2.rg_exc_bursting, R_rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(R_rg2.rg_exc_bursting, R_rg1.rg_exc_tonic,'custom_rg_rg')
    conn.create_connections(R_rg2.rg_exc_tonic, R_rg1.rg_exc_bursting,'custom_rg_rg')
    conn.create_connections(R_rg2.rg_exc_tonic, R_rg1.rg_exc_tonic,'custom_rg_rg')


    # =======================================================
    # V0D CONTRALATERAL CODE BELOW 
    # V0d bursting at low locomotion and tonic at high​ locomotion 

    # create contralateral interneuron populations - BURSTING AND TONIC SUBTYPES 
    R_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='tonic', pop_size=nn.v0d_pop_size, input_type='none') 
    R_V0D.create_commissural_population(pop_type='V0D', self_connection='none', firing_behavior='bursting', pop_size=nn.v0d_pop_size, input_type='none') 

    if nn.low_locomotion_v0d_right == 1: 
        
        print("Creating Connections for RIGHT V0d - LOW LOCOMOTION")

        # burst connections v0d ipsilateral from both rg bursting and exc

        # low locomotion burst ipsilateral connection - RG 1 FLEX  to R_V0D excitatory connection. 
        conn.create_connections(R_rg1.rg_exc_bursting, R_V0D.v0d_bursting, 'custom_rg_v0d_R')

        # conn.create_connections(R_rg1.rg_inh_bursting, R_V0D.v0d_bursting, 'custom_rg_v0d')
         
        # bursting v0d to contralateral rg bursting - INHIBITORY CONNECTION
        conn.create_connections(R_V0D.v0d_bursting, L_rg1.rg_exc_bursting, 'custom_v0d_rg_inh_R')

    if nn.high_locomotion_v0d_right == 1: 
        
        print("Creating Connections for RIGHT V0d - HIGH LOCOMOTION")

        # tonic connections v0d ipsilateral from both rg bursting and exc
        conn.create_connections(R_rg1.rg_exc_tonic, R_V0D.v0d_tonic, 'custom_rg_v0d_R')

        conn.create_connections(R_rg1.rg_inh_tonic, R_V0D.v0d_tonic, 'custom_rg_v0d_inh_R')

        # tonic v0d to contralateral rg tonic 
        conn.create_connections(R_V0D.v0d_tonic, L_rg1.rg_exc_tonic, 'custom_v0d_rg_inh_R')

     
    # =======================================================
    # V0V CONTRALATERAL CODE BELOW 
    # V0v bursting at high locomotion and tonic at low locomotion 
    
    R_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='descending') 
    
    R_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='bursting', pop_size=nn.v0v_pop_size, input_type='descending') 
    
    if nn.low_locomotion_v0v_right == 1: 

        print("Creating connections for RIGHT V0v - LOW LOCOMOTION ")

        # CONNECTION UNO: L_rg1 tonic input to L_V2a tonic. 
        # Here in certain connections they overlap with the previous unilateral architecture therefore we design weights XXX_v0vconn so we have a difference. 

        conn.create_connections(R_rg1.rg_exc_tonic, R_exc1.exc_inter_tonic, 'custom_rg_v2a_v0vconn')
        
        # tonic V0V receives tonic V2a (low frequency mode)
        conn.create_connections(R_exc1.exc_inter_tonic, R_V0V.v0v_tonic, 'custom_v2a_v0v')

        # tonic V0V excites V1 
        conn.create_connections(R_V0V.v0v_tonic, R_inh2.inh_inter_tonic, 'custom_v0v_v1')

        # V1 inhibits the RG_flx 
        conn.create_connections(R_inh2.inh_inter_tonic, L_rg1.rg_exc_tonic,  'custom_v1_rg_v0vconn')


    if nn.high_locomotion_v0v_right == 1: 

        print("Creating connections for RIGHT V0v - HIGH LOCOMOTION")


                # L_rg1 bursting input to L_V2a bursting. 
        conn.create_connections(R_rg1.rg_exc_bursting, R_exc1.exc_inter_burst, 'custom_rg_v2a')
        
        # bursting V0V receives tonic V2a (low frequency mode)
        conn.create_connections(R_exc1.exc_inter_burst, R_V0V.v0v_bursting, 'custom_v2a_v0v')

        # tonic V0V excites V1 
        conn.create_connections(R_V0V.v0v_bursting, R_inh2.inh_inter_bursting, 'custom_v0v_v1')

        # V1 inhibits the RG_flx 
        conn.create(R_inh2.inh_inter_bursting, L_rg1.rg_exc_bursting,  'custom_v1_rg_contra')

    # Defining V0c subpopulations have already been defined... 
    
    if nn.contralateral_projections_v0c_right == 1: 

        # 1 = Flx, 2 = Ext.
        # flexing V0c goes to flx Right MNP - exciting it 
        # extensing V0c goes to the ext Right MNP - exciting it  

        conn.create_connections(R_V0C_1.v0c_tonic, L_mnp1.motor_neuron_pop, 'custom_v0c_mnp_flx')
        conn.create_connections(R_V0C_2.v0c_tonic, L_mnp2.motor_neuron_pop, 'custom_v0c_mnp_ext')

    # ================================ END OF RIGHT SIDE CPG CODE ==========================================================
   
    ##  =============================== ONLINE RAMP =================================
    if nn.online_ramp_experiment == 1:

        print("[INFO] Online Ramp Experiment is ON")
        print("scale_start =", nn.w_start, "scale_end =", nn.w_end)

        ramp_duration = float(nn.ramp_duration)  # ms

        online_ramp_log = {"time": [], "weight": []}

        specified_weight = nn.online_ramp_weight  # e.g. "custom_v0d_rg_inh_L"
        connections = conn.synapses[specified_weight]
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
            print("[WARN] Mean weight is positive. This pathway may not be net-inhibitory.")


        # init sim
        init_time = 50.0
        nest.Simulate(init_time)
        t0 = float(nest.biological_time)

        step_ms = float(nn.time_resolution) * 10.0
        num_steps = int(float(nn.block_time) / step_ms)

        t_start = time.perf_counter()

        for _ in range(num_steps):

            nest.Simulate(step_ms)
            t = float(nest.biological_time)

            if ramp_duration > 0:
                prog = (t - t0) / ramp_duration
            else:
                prog = 1.0
            prog = max(0.0, min(1.0, prog))

            scale = float(nn.w_start + (nn.w_end - nn.w_start) * prog)

            # APPLY scaled weights (per-connection set)
            w_new = w0 * scale
            nest.SetStatus(connections, [{"weight": float(w)} for w in w_new])

            online_ramp_log["time"].append(t)
            online_ramp_log["weight"].append(scale)  # log scale factor

            # ---- feedback block unchanged (same as above) ----
            # (keep your existing feedback code here)

            print(f"t = {t:.1f} ms, {specified_weight} scale: {scale:.3f}", end="\r")

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
            L_V0V, R_V0V,
            L_V0D, R_V0D,
            L_label, 
            online_ramp_log, # we passing dict for online experiment. dict =  time & applied weight. 
            nn.online_ramp_weight,
            ramp_type="online"
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
            R_V0V, L_V0V,
            R_V0D, L_V0D,
            R_label,
            online_ramp_log, # we passing dict for online experiment. dict =  time & applied weight. 
            nn.online_ramp_weight,
            ramp_type="online"
            )
         ##  =============================== END OF ONLINE RAMP ===============================

    # ================= OFFLINE BLOCK STEPWISE RAMP =================
    if nn.offline_ramp_experiment == 1:

        print("[INFO] OFFLINE Ramp Experiment is ON")

        scales = [0, 2, 4, 8]  # consider [1,2,4,8] if you want baseline included

        specified_weight = nn.offline_ramp_weight
        connections = conn.synapses[specified_weight]
        print(f"[INFO] Ramp target: {specified_weight}, n_conns={len(connections)}")

        # Snapshot baseline distribution ONCE
        w0 = np.asarray(nest.GetStatus(connections, "weight"), dtype=float)
        print("[BASELINE weights]")
        print("  mean:", float(np.mean(w0)), "min:", float(np.min(w0)), "max:", float(np.max(w0)))

        init_time = 50.0
        nest.Simulate(init_time)

        block_duration = float(nn.block_time) / len(scales)
        step_ms = float(nn.time_resolution) * 10.0
        n_iters = int(block_duration / step_ms)

        for step_idx, s in enumerate(scales):

            # preserve distribution
            w_new = w0 * float(s)
            nest.SetStatus(connections, [{"weight": float(w)} for w in w_new])

            print(f"\n[OFFLINE RAMP] Block {step_idx+1}/{len(scales)} | scale={s} | applied_weight={mean_w:.4f}")

            t_start = time.perf_counter()

            for _ in range(n_iters):
                nest.Simulate(step_ms)

                # --- L side feedback ---
                num_spikes_flx_L = popfunc.read_recent_spike_data(L_mnp1.spike_detector_motor)
                num_spikes_ext_L = popfunc.read_recent_spike_data(L_mnp2.spike_detector_motor)
                L_fb.send_muscle_activation(num_spikes_flx_L, num_spikes_ext_L)
                flx_1a_L, ext_1a_L, flx_1b_L, ext_1b_L, *_ = L_fb.receive_muscle_afferents()

                if nn.fb_rg_flx: nest.SetStatus(L_rg1.rg_flx_pg, {"rate": flx_1a_L})
                if nn.fb_rg_ext: nest.SetStatus(L_rg2.rg_ext_pg, {"rate": ext_1b_L})
                if nn.fb_v2b:    nest.SetStatus(L_inh1.v2b_pg, {"rate": flx_1a_L})
                if nn.fb_v1:     nest.SetStatus(L_inh2.v1_pg, {"rate": ext_1b_L})

                # --- R side feedback ---
                num_spikes_flx_R = popfunc.read_recent_spike_data(R_mnp1.spike_detector_motor)
                num_spikes_ext_R = popfunc.read_recent_spike_data(R_mnp2.spike_detector_motor)
                R_fb.send_muscle_activation(num_spikes_flx_R, num_spikes_ext_R)
                flx_1a_R, ext_1a_R, flx_1b_R, ext_1b_R, *_ = R_fb.receive_muscle_afferents()

                if nn.fb_rg_flx: nest.SetStatus(R_rg1.rg_flx_pg, {"rate": flx_1a_R})
                if nn.fb_rg_ext: nest.SetStatus(R_rg2.rg_ext_pg, {"rate": ext_1b_R})
                if nn.fb_v2b:    nest.SetStatus(R_inh1.v2b_pg, {"rate": flx_1a_R})
                if nn.fb_v1:     nest.SetStatus(R_inh2.v1_pg, {"rate": ext_1b_R})

            t_stop = time.perf_counter()
            print(f"[INFO] Block sim complete in {t_stop - t_start:.2f} s | t={float(nest.biological_time):.1f} ms")

            # ---- run analysis/plots for this block ----
            L_label = f"LEFT_OFFLINE_x{s}"
            R_label = f"RIGHT_OFFLINE_x{s}"

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
                L_V0V, R_V0V,
                L_V0D, R_V0D,
                L_label,
                mean_w,                 # reportable scalar weight value
                specified_weight,       # correct weight name
                ramp_type="offline"
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
                R_V0V, L_V0V,
                R_V0D, L_V0D,
                R_label,
                mean_w,
                specified_weight,
                ramp_type="offline"
            )

                ##  =============================== END OF OFFLINE BLOCK STEPWISE RAMP ===============================

                
# if ramp experiments are not activated from the config file run simulations normally. 
if (nn.offline_ramp_experiment == 0) and (nn.online_ramp_experiment == 0):


    # ================= SIMULATION FOR BOTH L_ AND R_ =================

    print("Offline Ramp Parameter is off.")
    print("Seed#: ", nn.rng_seed)
    print("RG Flx: # exc (bursting, tonic): ", nn.flx_exc_bursting_count, nn.flx_exc_tonic_count,
        "; # inh(bursting, tonic): ", nn.flx_inh_bursting_count, nn.flx_inh_tonic_count)
    print("RG Ext: # exc (bursting, tonic): ", nn.ext_exc_bursting_count, nn.ext_exc_tonic_count,
        "; # inh(bursting, tonic): ", nn.ext_inh_bursting_count, nn.ext_inh_tonic_count)
    print("V2b/V1: # inh (tonic): ", nn.num_inh_inter_tonic_v2b, nn.num_inh_inter_tonic_v1)
    print("V2a: # exc (tonic): ", nn.v2a_tonic_pop_size, "; # MNs: ", nn.num_motor_neurons)

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
        L_inh1, L_inh2, R_inh2,
        L_rc_1, L_rc_2,
        L_mnp1, L_mnp2, R_mnp1, R_mnp2,
        L_V0V, R_V0V,
        L_V0D, R_V0D,
        L_label
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
        R_V0V, L_V0V,
        R_V0D, L_V0D,
        R_label
    )