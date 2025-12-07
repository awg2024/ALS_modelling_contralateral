
import time
import matplotlib.pyplot as plt
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


# =====================================================================


def build_full_cpg_network(): 

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

            # RG_F → V0d_bursting  (OK)
            conn.create_connections(L_rg1.rg_exc_bursting, L_V0D.v0d_bursting, 'custom_rg_v0d')

            # RG_E → V0d_bursting  (OK if you want both F/E to drive it)
            conn.create_connections(L_rg2.rg_inh_bursting, L_V0D.v0d_bursting, 'custom_rg_v0d_inh')

            # V0d_bursting → RG_F CONTRA (inhibitory, correct)
            conn.create_connections(L_V0D.v0d_bursting, R_rg1.rg_exc_bursting, 'custom_v0d_rg_inh')

        if nn.high_locomotion_v0d_left == 1: 
            
            print("Creating Connections for LEFT V0d - HIGH LOCOMOTION")

            # tonic connections v0d ipsilateral from both rg bursting and exc
            conn.create_connections(L_rg1.rg_exc_tonic, L_V0D.v0d_tonic, 'custom_rg_v0d')
            conn.create_connections(L_rg2.rg_inh_tonic, L_V0D.v0d_tonic, 'custom_rg_v0d_inh')

            # tonic v0d to contralateral rg tonic 
            conn.create_connections(L_V0D.v0d_tonic, R_rg1.rg_exc_tonic, 'custom_v0d_rg_inh')

    

        # =======================================================
        # V0V CONTRALATERAL CODE BELOW
        # V0v bursting at high locomotion and tonic at low locomotion 
        
        L_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='descending') 
        L_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='bursting', pop_size=nn.v0v_pop_size, input_type='descending') 
        
        if nn.low_locomotion_v0v_left == 1: 

            print("Creating connections for LEFT V0v - LOW LOCOMOTION ")
            
            conn.create_connections(L_rg1.rg_exc_tonic, L_exc1.exc_inter_tonic, 'custom_rg_v2a')
            # tonic V0V receives tonic V2a (low frequency mode)
            conn.create_connections(L_exc1.exc_inter_tonic, L_V0V.v0v_tonic, 'custom_v2a_v0v')

            # tonic V0V inhibits contralateral bursting RG (low frequency)
            conn.create_connections(L_V0V.v0v_tonic, R_rg1.rg_exc_bursting, 'custom_v0v_rg_inh')

        
        if nn.high_locomotion_v0v_left == 1: 

            print("Creating connections for LEFT V0v - HIGH LOCOMOTION")

            # bursting V0V receives bursting V2a (high frequency mode)
            conn.create_connections(L_exc1_burst.interneuron_pop, L_V0V.v0v_bursting, 'custom_v2a_v0v')

            # bursting V0V inhibits contralateral tonic RG (high frequency)
            conn.create_connections(L_V0V.v0v_bursting, R_rg1.rg_exc_tonic, 'custom_v0v_rg_inh')

    
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


        # 1 - FLX
        # 2 - EXT
        L_V0C_1.create_commissural_population(pop_type='V0C', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='none')
        L_V0C_2.create_commissural_population(pop_type='V0C', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='none')
        
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
            conn.create_connections(R_rg1.rg_exc_bursting, R_V0D.v0d_bursting, 'custom_rg_v0d')
            conn.create_connections(R_rg1.rg_inh_bursting, R_V0D.v0d_bursting, 'custom_rg_v0d')
            
            # bursting v0d to contralateral rg bursting - INHIBITORY CONNECTION
            conn.create_connections(R_V0D.v0d_bursting, L_rg1.rg_exc_bursting, 'custom_v0d_rg_inh')

        if nn.high_locomotion_v0d_right == 1: 
            
            print("Creating Connections for RIGHT V0d - HIGH LOCOMOTION")

            # tonic connections v0d ipsilateral from both rg bursting and exc
            conn.create_connections(R_rg1.rg_exc_tonic, R_V0D.v0d_tonic, 'custom_rg_v0d')
            conn.create_connections(R_rg1.rg_inh_tonic, R_V0D.v0d_tonic, 'custom_rg_v0d_inh')

            # tonic v0d to contralateral rg tonic 
            conn.create_connections(R_V0D.v0d_tonic, L_rg1.rg_exc_tonic, 'custom_v0d_rg_inh')

        
        # =======================================================
        # V0V CONTRALATERAL CODE BELOW 
        # V0v bursting at high locomotion and tonic at low locomotion 
        
        
        R_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='descending') 
        R_V0V.create_commissural_population(pop_type='V0V', self_connection='none', firing_behavior='bursting', pop_size=nn.v0v_pop_size, input_type='descending') 
        
        if nn.low_locomotion_v0v_right == 1: 

            print("Creating connections for RIGHT V0v - LOW LOCOMOTION ")

            conn.create_connections(R_rg1.rg_exc_tonic, R_exc1.exc_inter_tonic, 'custom_rg_v2a')

            # tonic V0V receives tonic V2a (low frequency mode)
            conn.create_connections(R_exc1.exc_inter_tonic, R_V0V.v0v_tonic, 'custom_v2a_v0v')

            # tonic V0V inhibits contralateral bursting RG (low frequency)
            conn.create_connections(R_V0V.v0v_tonic, L_rg1.rg_exc_bursting, 'custom_v0v_rg_inh')

        if nn.high_locomotion_v0v_right == 1: 

            print("Creating connections for RIGHT V0v - HIGH LOCOMOTION")


            # bursting V0V receives tonic V2a (high frequency mode)
            conn.create_connections(R_exc1_burst.interneuron_pop, R_V0V.v0v_bursting, 'custom_v2a_v0v')

            # bursting V0V inhibits contralateral tonic RG (high frequency)
            conn.create_connections(R_V0V.v0v_bursting, R_exc1_burst.interneuron_pop, 'custom_v0v_rg_inh')
        
        # 1 - FLX
        # 2 - EXT
        R_V0C_1.create_commissural_population(pop_type='V0C', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='none')
        R_V0C_2.create_commissural_population(pop_type='V0C', self_connection='none', firing_behavior='tonic', pop_size=nn.v0v_pop_size, input_type='none')
        
        if nn.contralateral_projections_v0c_right == 1: 

            # 1 = Flx, 2 = Ext.
            # flexing V0c goes to flx Right MNP - exciting it 
            # extensing V0c goes to the ext Right MNP - exciting it  

            conn.create_connections(R_V0C_1.v0c_tonic, L_mnp1.motor_neuron_pop, 'custom_v0c_mnp_flx')
            conn.create_connections(R_V0C_2.v0c_tonic, L_mnp2.motor_neuron_pop, 'custom_v0c_mnp_ext')

    pops = {
        "L_mnp1": L_mnp1,
        "L_mnp2": L_mnp2,
        "R_mnp1": R_mnp1,
        "R_mnp2": R_mnp2,
        "L_rg1": L_rg1,
        "L_rg2": L_rg2,
        "R_rg1": R_rg1,
        "R_rg2": R_rg2,
        "L_V0D": L_V0D,
        "R_V0D": R_V0D,
        "L_V0V": L_V0V,
        "R_V0V": R_V0V,
        "L_exc1": L_exc1,
        "L_exc2": L_exc2,
        "R_exc1": R_exc1,
        "R_exc2": R_exc2,
        "L_V1a_1": L_V1a_1,
        "L_V1a_2": L_V1a_2,
        "R_V1a_1": R_V1a_1,
        "R_V1a_2": R_V1a_2,
        # etc ... you can add all populations you need later
    }
    
    return pops


if __name__ == "__main__":

# ==========================================================================
#  THIS IS TESTING_THREAD SPEED OF THE CPU FOR ANALYSIS 
# ==========================================================================

    nn = netparams.neural_network()   # keep time resolution + RNG seed consistent

    thread_values = [2, 4, 6, 8, 10, 12, 16, 20, 24]   # edit depending on your CPU
    times = []

    print("\n[THREAD BENCHMARK] Running short NEST performance test...\n")

    for t in thread_values:
        # Reset simulator before testing thread count
        nest.ResetKernel()

        nest.SetKernelStatus({
            "local_num_threads": t,
            "resolution": nn.time_resolution,
            "rng_seed": nn.rng_seed,
        })

        # Start kernel + build network
        ss.nest_start()
        build_full_cpg_network()

        # Time 500 ms of simulation
        t0 = time.time()
        nest.Simulate(500.0)
        t1 = time.time()

        wall_time = t1 - t0
        times.append(wall_time)
        print(f"Threads={t:2d} → {wall_time:.2f} sec")

    # ---- Plot scaling curve ----
    plt.figure(figsize=(8,4))
    plt.plot(thread_values, times, marker="o", linewidth=2)
    plt.xlabel("local_num_threads")
    plt.ylabel("Wall-clock time (s) for 500 ms simulation")
    plt.title("NEST Thread Scaling Benchmark")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n[✔] Benchmark complete.")
