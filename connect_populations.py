#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import time 
import pylab
import math
import matplotlib.pyplot as plt
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.signal import find_peaks, peak_widths, peak_prominences
import time
import numpy as np
import copy
from set_network_params import neural_network
netparams = neural_network()

def create_synapse_params(mean, std, delay):
    """
    Helper function to create synapse parameters with random weight and fixed delay.
    
    Args:
        mean (float): Mean of the weight distribution.
        std (float): Standard deviation of the weight distribution.
        delay (float): Synaptic delay.
    
    Returns:
        dict: Synapse parameters including weight and delay.
    """
    return {
        "synapse_model": "static_synapse",
        "weight": nest.random.normal(mean=mean, std=std),  # nS
        "delay": delay  # ms
    }

class ConnectNetwork():
    def __init__(self):
        self.total_weight_exc = 0
        self.total_weight_inh = 0
        self.balance_pct = 0
        self.num_of_synapses = []
        self.name_of_pops = []
        self.used_synapse_keys = set()
        self.synapses = {}

        self.synapse_params = {

            # SYMMETRIC CONNECTIONS 


            'custom_rg_v1': {
                'conn_dict': 'conn_dict_custom_rg_v1',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v1_mean, netparams.w_custom_rg_v1_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v1',
                'conn_type': 'rg_layer'
            },
            'custom_rg_v2b': {
                'conn_dict': 'conn_dict_custom_rg_v2b',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2b_mean, netparams.w_custom_rg_v2b_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2b'
            },
            'custom_rg_rg': {
                'conn_dict': 'conn_dict_custom_rg1_rg2',
                'syn_params': create_synapse_params(netparams.w_custom_rg_rg_mean, netparams.w_custom_rg_rg_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_rg'
            },
            'custom_rg_v1a': {
                'conn_dict': 'conn_dict_custom_rg_v1a',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v1a_mean, netparams.w_custom_rg_v1a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v1a'
            },
            'custom_mn_rc': {
                'conn_dict': 'conn_dict_custom_mn_rc',
                'syn_params': create_synapse_params(netparams.w_custom_mn_rc_mean, netparams.w_custom_mn_rc_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_mn_rc'
            },
            'custom_rc_rc': {
                'conn_dict': 'conn_dict_custom_rc_rc',
                'syn_params': create_synapse_params(netparams.w_custom_rc_rc_mean, netparams.w_custom_rc_rc_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rc_rc'
            },
            'custom_rc_v1a': {
                'conn_dict': 'conn_dict_custom_rc_v1a',
                'syn_params': create_synapse_params(netparams.w_custom_rc_v1a_mean, netparams.w_custom_rc_v1a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rc_v1a'
            },
            'custom_rc_mn': {
                'conn_dict': 'conn_dict_custom_rc_mn',
                'syn_params': create_synapse_params(netparams.w_custom_rc_mn_mean, netparams.w_custom_rc_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rc_mn'
            },
            'custom_v1a_mn': {
                'conn_dict': 'conn_dict_custom_v1a_mn',
                'syn_params': create_synapse_params(netparams.w_custom_v1a_mn_mean, netparams.w_custom_v1a_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1a_mn'
            },
            'custom_v1a_v1a': {
                'conn_dict': 'conn_dict_custom_v1a_v1a',
                'syn_params': create_synapse_params(netparams.w_custom_v1a_v1a_mean, netparams.w_custom_v1a_v1a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1a_v1a'
            },
            'custom_v1_rg': {
                'conn_dict': 'conn_dict_custom_v1_rg',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_mean, netparams.w_custom_v1_rg_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg'
            },
            'custom_v2b_rg': {
                'conn_dict': 'conn_dict_custom_v2b_rg',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_rg_mean, netparams.w_custom_v2b_rg_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_rg'
            },
            'custom_v1_v2a': {
                'conn_dict': 'conn_dict_custom_v1_v2a',
                'syn_params': create_synapse_params(netparams.w_custom_v1_v2a_mean, netparams.w_custom_v1_v2a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_v2a'
            },
            'custom_v2b_v2a': {
                'conn_dict': 'conn_dict_custom_v2b_v2a',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_v2a_mean, netparams.w_custom_v2b_v2a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_v2a'
            },
            'custom_v0c_mn': {
                'conn_dict': 'conn_dict_custom_v0c_mn',
                'syn_params': create_synapse_params(netparams.w_custom_v0c_mn_mean, netparams.w_custom_v0c_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0c_mn'
            },
            'custom_rg_v2a': { 
                'conn_dict': 'conn_dict_custom_rg_v2a',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2a_mean, netparams.w_custom_rg_v2a_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2a'
            },
            'custom_rg_v0c': {
                'conn_dict': 'conn_dict_custom_rg_v0c',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v0c_mean, netparams.w_custom_rg_v0c_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v0c'
            },
            'custom_v2a_mn': {
                'conn_dict': 'conn_dict_custom_v2a_mn',
                'syn_params': create_synapse_params(netparams.w_custom_v2a_mn_mean, netparams.w_custom_v2a_mn_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2a_mn'
            },
             'custom_rg_v0d': { #COMMISSURAL V0D 
                'conn_dict': 'conn_dict_custom_rg_v0d',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v0d_mean, netparams.w_custom_rg_v0d_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v0d'
            },
            'custom_v0v_rg_inh': { #COMMISSURAL V0V
                'conn_dict': 'conn_dict_custom_v0v_rg_inh',
                'syn_params': create_synapse_params(netparams.w_custom_v0v_rg_inh_mean, netparams.w_custom_v0v_rg_inh_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0v_rg_inh'
            },
            'custom_v2a_v0v': { #COMMISSURAL V0V
                'conn_dict': 'conn_dict_custom_v2a_v0v',
                'syn_params': create_synapse_params(netparams.w_custom_v2a_v0v_mean, netparams.w_custom_v2a_v0v_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2a_v0v'
            },
            'custom_v0c_mnp_flx': {
                'conn_dict': 'conn_dict_custom_v0c_mnp_flx',
                'syn_params': create_synapse_params(netparams.w_custom_v0c_mnp_flx_mean, netparams.w_custom_v0c_mnp_flx_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0c_mnp_flx'
            },
            'custom_v0c_mnp_ext': {
                'conn_dict': 'conn_dict_custom_v0c_mnp_ext',
                'syn_params': create_synapse_params(netparams.w_custom_v0c_mnp_ext_mean, netparams.w_custom_v0c_mnp_ext_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0c_mnp_ext'
            },
            'custom_v0v_v1': { #COMMISSURAL V0V
                'conn_dict': 'conn_dict_custom_v0v_v1',
                'syn_params': create_synapse_params(netparams.w_custom_v0v_v1_mean, netparams.w_custom_v0v_v1_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0v_v1'
            },
            'custom_rg_v2a_v0v_L': {
                'conn_dict': 'conn_dict_custom_rg_v2a_v0v_L', 
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2a_v0v_L_mean, netparams.w_custom_rg_v2a_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2a_v0v_L'
            },
            'custom_v1_rg_v0v': {
                'conn_dict': 'conn_dict_custom_v1_rg_v0vconn',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_v0v_mean, netparams.w_custom_v1_rg_v0v_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg_v0v'
            },
            'custom_v2a_v0v_L': {
                'conn_dict': 'conn_dict_custom_v2a_v0v_L',
                'syn_params': create_synapse_params(netparams.w_custom_v2a_v0v_L_mean, netparams.w_custom_v2a_v0v_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2a_v0v_L'
            },
            'custom_v0v_v1_L': {
                'conn_dict': 'conn_dict_custom_v0v_v1_L',
                'syn_params': create_synapse_params(netparams.w_custom_v0v_v1_L_mean, netparams.w_custom_v0v_v1_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0v_v1_L'
            },
            'custom_v1_rg_v0v_L': {
                'conn_dict': 'conn_dict_custom_v1_rg_v0v_L',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_v0v_L_mean, netparams.w_custom_v1_rg_v0v_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg_v0v_L'
            },
            # ======================================================================================================================================================
            # ASYMMETRIC CONNECTIONS 

            'custom_rg_v0d_R': { #COMMISSURAL V0D 
                'conn_dict': 'conn_dict_custom_rg_v0d_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v0d_R_mean, netparams.w_custom_rg_v0d_R_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v0d_R'
            },
            'custom_rg_v0d_L': { #COMMISSURAL V0D 
                'conn_dict': 'conn_dict_custom_rg_v0d_L',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v0d_L_mean, netparams.w_custom_rg_v0d_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v0d_L'
            },

            'custom_v0d_rg_inh_L': { # LEFT COMMISSURAL CONNECTION V0D
                'conn_dict': 'conn_dict_custom_v0d_rg_inh_L',
                'syn_params': create_synapse_params(netparams.w_custom_v0d_rg_inh_L_mean, netparams.w_custom_v0d_rg_inh_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0d_rg_inh_L'
            },
            'custom_v0d_rg_inh_R': { # RIGHT COMMISSURAL CONNECTION V0D
                'conn_dict': 'conn_dict_custom_v0d_rg_inh_R',
                'syn_params': create_synapse_params(netparams.w_custom_v0d_rg_inh_R_mean, netparams.w_custom_v0d_rg_inh_R_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0d_rg_inh_R'
            },
            'custom_v1_rg_L': {
                'conn_dict': 'conn_dict_custom_v1_rg_L',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_L_mean, netparams.w_custom_v1_rg_L_std, netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg_L'
            },

          
            # ---- RC -> MN (L/R) ----
            'custom_rc_mn_L': {
                'conn_dict': 'conn_dict_custom_rc_mn_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rc_mn_L_mean,
                    netparams.w_custom_rc_mn_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rc_mn_L'
            },
            'custom_rc_mn_R': {
                'conn_dict': 'conn_dict_custom_rc_mn_R',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rc_mn_R_mean,
                    netparams.w_custom_rc_mn_R_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rc_mn_R'
            },

            # ---- V1a -> MN (L/R) ----
            'custom_v1a_mn_L': {
                'conn_dict': 'conn_dict_custom_v1a_mn_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v1a_mn_L_mean,
                    netparams.w_custom_v1a_mn_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v1a_mn_L'
            },

            'custom_v1a_mn_R': {
                'conn_dict': 'conn_dict_custom_v1a_mn_R',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v1a_mn_R_mean,
                    netparams.w_custom_v1a_mn_R_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v1a_mn_R'
            },

            # ---- RC -> V1a (L/R) ----
            'custom_rc_v1a_L': {
                'conn_dict': 'conn_dict_custom_rc_v1a_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rc_v1a_L_mean,
                    netparams.w_custom_rc_v1a_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rc_v1a_L'
            },

            'custom_rc_v1a_R': {
                'conn_dict': 'conn_dict_custom_rc_v1a_R',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rc_v1a_R_mean,
                    netparams.w_custom_rc_v1a_R_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rc_v1a_R'
            },
            # ---- RC -> RC (L/R) ----
            'custom_rc_rc_L': {
                'conn_dict': 'conn_dict_custom_rc_rc_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rc_rc_L_mean,
                    netparams.w_custom_rc_rc_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rc_rc_L'
            },

            'custom_rc_rc_R': {
                'conn_dict': 'conn_dict_custom_rc_rc_R',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rc_rc_R_mean,
                    netparams.w_custom_rc_rc_R_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rc_rc_R'
            },

            # ---- V1a -> V1a (L/R) ----
            'custom_v1a_v1a_L': {
                'conn_dict': 'conn_dict_custom_v1a_v1a_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v1a_v1a_L_mean,
                    netparams.w_custom_v1a_v1a_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v1a_v1a_L'
            },

            'custom_v1a_v1a_R': {
                'conn_dict': 'conn_dict_custom_v1a_v1a_R',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v1a_v1a_R_mean,
                    netparams.w_custom_v1a_v1a_R_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v1a_v1a_R'
            },
            'custom_rg_v2a_L': {
                'conn_dict': 'conn_dict_custom_rg_v2a_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rg_v2a_L_mean,
                    netparams.w_custom_rg_v2a_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rg_v2a_L'
            },
            'custom_v2a_mn_L': {
                'conn_dict': 'conn_dict_custom_v2a_mn_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v2a_mn_L_mean,
                    netparams.w_custom_v2a_mn_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v2a_mn_L'
            },
            'custom_rg_v1a_L': {
                'conn_dict': 'conn_dict_custom_rg_v1a_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rg_v1a_L_mean,
                    netparams.w_custom_rg_v1a_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rg_v1a_L'
            },
            'custom_rg_v0c_L': {
                'conn_dict': 'conn_dict_custom_rg_v0c_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rg_v0c_L_mean,
                    netparams.w_custom_rg_v0c_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rg_v0c_L'
            },
            'custom_v0c_mn_L': {
                'conn_dict': 'conn_dict_custom_v0c_mn_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v0c_mn_L_mean,
                    netparams.w_custom_v0c_mn_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v0c_mn_L'
            },
            'custom_rg_v2b_L': {
                'conn_dict': 'conn_dict_custom_rg_v2b_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rg_v2b_L_mean,
                    netparams.w_custom_rg_v2b_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rg_v2b_L'
            },
            'custom_rg_v1_L': {
                'conn_dict': 'conn_dict_custom_rg_v1_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rg_v1_L_mean,
                    netparams.w_custom_rg_v1_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rg_v1_L'
            },
            'custom_v2b_rg_L': {
                'conn_dict': 'conn_dict_custom_v2b_rg_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v2b_rg_L_mean,
                    netparams.w_custom_v2b_rg_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v2b_rg_L'
            },
            'custom_v2b_v2a_L': {
                'conn_dict': 'conn_dict_custom_v2b_v2a_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_v2b_v2a_L_mean,
                    netparams.w_custom_v2b_v2a_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_v2b_v2a_L'
            },
            'custom_rg_rg_L': {
                'conn_dict': 'conn_dict_custom_rg_rg_L',
                'syn_params': create_synapse_params(
                    netparams.w_custom_rg_rg_L_mean,
                    netparams.w_custom_rg_rg_L_std,
                    netparams.synaptic_delay
                ),
                'sparsity': 'sparsity_custom_rg_rg_L'
            },
           
            'custom_v1_v2a_L': {
                'conn_dict': 'conn_dict_custom_v1_v2a_L',
                'syn_params': create_synapse_params(netparams.w_custom_v1_v2a_L_mean,
                                                    netparams.w_custom_v1_v2a_L_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_v2a_L'
            },

            'custom_v2b_mn_L': {
                'conn_dict': 'conn_dict_custom_v2b_mn_L',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_mn_L_mean,
                                                    netparams.w_custom_v2b_mn_L_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_mn_L'
            },

            'custom_v1_mn_L': {
                'conn_dict': 'conn_dict_custom_v1_mn_L',
                'syn_params': create_synapse_params(netparams.w_custom_v1_mn_L_mean,
                                                    netparams.w_custom_v1_mn_L_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_mn_L'
            },

            'custom_rg_v2a_v0v': {
                'conn_dict': 'conn_dict_custom_rg_v2a_v0v',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2a_v0v_mean,
                                                    netparams.w_custom_rg_v2a_v0v_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2a_v0v'
            },

            'custom_v0v_v1': {
                'conn_dict': 'conn_dict_custom_v0v_v1',
                'syn_params': create_synapse_params(netparams.w_custom_v0v_v1_mean,
                                                    netparams.w_custom_v0v_v1_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0v_v1'
            },

            'custom_v1_rg_v0v': {
                'conn_dict': 'conn_dict_custom_v1_rg_v0v',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_v0v_mean,
                                                    netparams.w_custom_v1_rg_v0v_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg_v0v'
            },

            'custom_rg_v1_R': {
                'conn_dict': 'conn_dict_custom_rg_v1_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v1_R_mean,
                                                    netparams.w_custom_rg_v1_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v1_R'
            },

            'custom_v2b_rg_R': {
                'conn_dict': 'conn_dict_custom_v2b_rg_R',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_rg_R_mean,
                                                    netparams.w_custom_v2b_rg_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_rg_R'
            },

            'custom_v1_rg_R': {
                'conn_dict': 'conn_dict_custom_v1_rg_R',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_R_mean,
                                                    netparams.w_custom_v1_rg_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg_R'
            },

            'custom_v2b_v2a_R': {
                'conn_dict': 'conn_dict_custom_v2b_v2a_R',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_v2a_R_mean,
                                                    netparams.w_custom_v2b_v2a_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_v2a_R'
            },

            'custom_v1_v2a_R': {
                'conn_dict': 'conn_dict_custom_v1_v2a_R',
                'syn_params': create_synapse_params(netparams.w_custom_v1_v2a_R_mean,
                                                    netparams.w_custom_v1_v2a_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_v2a_R'
            },

            'custom_v2b_mn_R': {
                'conn_dict': 'conn_dict_custom_v2b_mn_R',
                'syn_params': create_synapse_params(netparams.w_custom_v2b_mn_R_mean,
                                                    netparams.w_custom_v2b_mn_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2b_mn_R'
            },

            'custom_v1_mn_R': {
                'conn_dict': 'conn_dict_custom_v1_mn_R',
                'syn_params': create_synapse_params(netparams.w_custom_v1_mn_R_mean,
                                                    netparams.w_custom_v1_mn_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_mn_R'
            },

            'custom_rg_rg_R': {
                'conn_dict': 'conn_dict_custom_rg_rg_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_rg_R_mean,
                                                    netparams.w_custom_rg_rg_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_rg_R'
            },
            'custom_rg_v2b_R' : {
                'conn_dict': 'conn_dict_custom_rg_v2b_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2b_R_mean,
                                                    netparams.w_custom_rg_v2b_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2b_R'
            },
            'custom_rg_v2a_R' : {
            'conn_dict': 'conn_dict_custom_rg_v2a_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2a_R_mean,
                                                    netparams.w_custom_rg_v2a_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2a_R'
            },
            'custom_v2a_mn_R' : {
            'conn_dict': 'conn_dict_custom_v2a_mn_R',
                'syn_params': create_synapse_params(netparams.w_custom_v2a_mn_R_mean,
                                                    netparams.w_custom_v2a_mn_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2a_mn_R'
            },
            'custom_rg_v1a_R' : {
            'conn_dict': 'conn_dict_custom_rg_v1a_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v1a_R_mean,
                                                    netparams.w_custom_rg_v1a_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v1a_R'
            },
            'custom_rg_v0c_R' : {
            'conn_dict': 'conn_dict_custom_rg_v0c_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v0c_R_mean,
                                                    netparams.w_custom_rg_v0c_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v0c_R'
            },
            'custom_v0c_mn_R' : {
            'conn_dict': 'conn_dict_custom_v0c_mn_R',
                'syn_params': create_synapse_params(netparams.w_custom_v0c_mn_R_mean,
                                                    netparams.w_custom_v0c_mn_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0c_mn_R'
            },
            'custom_v1a_v1a_R' : {
            'conn_dict': 'conn_dict_custom_v1a_v1a_R',
                'syn_params': create_synapse_params(netparams.w_custom_v1a_v1a_R_mean,
                                                    netparams.w_custom_v1a_v1a_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1a_v1a_R'
            },
            'custom_rg_v2a_v0v_R' : {
            'conn_dict': 'conn_dict_custom_rg_v2a_v0v_R',
                'syn_params': create_synapse_params(netparams.w_custom_rg_v2a_v0v_R_mean,
                                                    netparams.w_custom_rg_v2a_v0v_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_rg_v2a_v0v_R'
            },
            'custom_v2a_v0v_R' : {
            'conn_dict': 'conn_dict_custom_v2a_v0v_R',
                'syn_params': create_synapse_params(netparams.w_custom_v2a_v0v_R_mean,
                                                    netparams.w_custom_v2a_v0v_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v2a_v0v_R'
            },
            'custom_v0v_v1_R' : {
            'conn_dict': 'conn_dict_custom_v0v_v1_R',
                'syn_params': create_synapse_params(netparams.w_custom_v0v_v1_R_mean,
                                                    netparams.w_custom_v0v_v1_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v0v_v1_R'
            },
            'custom_v1_rg_v0v_R' : {
            'conn_dict': 'conn_dict_custom_v1_rg_v0v_R',
                'syn_params': create_synapse_params(netparams.w_custom_v1_rg_v0v_R_mean,
                                                    netparams.w_custom_v1_rg_v0v_R_std,
                                                    netparams.synaptic_delay),
                'sparsity': 'sparsity_custom_v1_rg_v0v_R'
            },
           
           # v3 connections low locomotion flx - flx. 
           'custom_rg_v3_L': {
            'conn_dict': 'conn_dict_custom_rg_v3_L',
            'syn_params':create_synapse_params(netparams.w_custom_rg_v3_L_mean,
                                                    netparams.w_custom_rg_v3_L_std,
                                                    netparams.synaptic_delay),
            'sparsity': 'sparsity_custom_rg_v3_L'    
            }, 
            'custom_v3_rg_L': {
            'conn_dict': 'conn_dict_custom_v3_rg_L',
            'syn_params':create_synapse_params(netparams.w_custom_v3_rg_L_mean,
                                                    netparams.w_custom_v3_rg_L_std,
                                                    netparams.synaptic_delay),
            'sparsity':'sparsity_custom_v3_rg_L'
            },
            'custom_rg_v3_R': {
            'conn_dict': 'conn_dict_custom_rg_v3_R',
            'syn_params':create_synapse_params(netparams.w_custom_rg_v3_R_mean,
                                                    netparams.w_custom_rg_v3_R_std,
                                                    netparams.synaptic_delay),
            'sparsity': 'sparsity_custom_rg_v3_R'
            },
            
            'custom_v3_rg_R': {
            'conn_dict': 'conn_dict_custom_v3_rg_R',
            'syn_params':create_synapse_params(netparams.w_custom_v3_rg_R_mean,
                                                    netparams.w_custom_v3_rg_R_std,
                                                    netparams.synaptic_delay),
            'sparsity': 'sparsity_custom_v3_rg_R'
            },

        }

    
    def create_connections(self, pop1, pop2, syn_type):
        """
        Creates connections between populations with specified synapse type.

        Stores SynapseCollections in self.synapses[syn_type] as a LIST so repeated calls
        for the same syn_type are accumulated (not overwritten).
        """
        # record that this synapse key was actually used
        self.used_synapse_keys.add(syn_type)

        # validate synapse key
        if syn_type not in self.synapse_params:
            print(f"Warning: Invalid synapse type: {syn_type}. Skipping.")
            return

        # fetch config
        conn_dict = getattr(netparams, self.synapse_params[syn_type]["conn_dict"])
        syn_params = self.synapse_params[syn_type]["syn_params"]

        if syn_type in ("custom_v0d_rg_inh_L", "custom_v0d_rg_inh_R"):
            print("[DEBUG PARAMS]", syn_type,
                "mean", getattr(netparams, f"w_{syn_type}_mean", "MISSING"),
                "std",  getattr(netparams, f"w_{syn_type}_std", "MISSING"),
                "syn_params.weight=", syn_params["weight"])

            time.sleep(20)

        sparsity   = getattr(netparams, self.synapse_params[syn_type]["sparsity"])

        # (optional) before/after count for this exact (src,tgt) pair
        before = len(nest.GetConnections(source=pop1, target=pop2))

        # create the connection
        nest.Connect(pop1, pop2, conn_dict, syn_params)

        # capture just-created (src,tgt) connection set
        conns = nest.GetConnections(source=pop1, target=pop2)

        # accumulate handles for later experiments/analysis
        self.synapses.setdefault(syn_type, []).append(conns)

        # stats / bookkeeping
        after = len(conns)
        added = after - before

        self.local_connections = after
        self.name_of_pops.append(syn_type)
        self.num_of_synapses.append(self.local_connections)

        weight_mean_name = f"w_{syn_type}_mean"
        weight_std_name  = f"w_{syn_type}_std"

        print(
            f"{syn_type} connection created, connectivity %, weight (mean,std) = {sparsity},",
            getattr(netparams, weight_mean_name),
            getattr(netparams, weight_std_name)
        )
        #print(f"[CONNECTION INFO] {syn_type}: before={before} after={after} added={added}")
        
    def calculate_synapse_percentage(self):
        self.all_connections = len(nest.GetConnections())
        self.percentage_of_connections = [x//self.all_connections for x in self.num_of_synapses]       
        print('Total connections: ',self.all_connections)
        print('Name of connections: ',self.name_of_pops)
        print('Local connections: ',self.num_of_synapses)
        
    def sum_weights_per_source(self,population):
        synapse_data = nest.GetConnections(population).get(['source', 'weight'])
        weights_per_source = {}
        for connection in synapse_data:
            source_neuron = synapse_data['source']
            weights = synapse_data['weight']
            for s in set(source_neuron):
                if s not in weights_per_source:
                    weights_per_source[s] = sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
                else:
                    weights_per_source[s] += sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
        return weights_per_source
    
    def count_spikes_per_source(self,spike_detector):
        sender_counts = {}
        spike_data = spike_detector.get('events', 'senders')
        #print('Sender data: ',spike_data)
        for sender_list in spike_data:
            for sender in sender_list:
                if sender not in sender_counts:
                    sender_counts[sender] = 1
                else:
                    sender_counts[sender] += 1
        return sender_counts
    
    def calculate_weighted_balance(self, pop1,spike_detector):
        self.total_weight = 0 
        self.weights_by_source = self.sum_weights_per_source(pop1)
        self.sender_counts = self.count_spikes_per_source(spike_detector)
        #print('Count per neuron ID: ',self.sender_counts)        
        #print('Weights by source: ',self.weights_by_source)
        for source in self.weights_by_source:
            #print('Neuron ID: ',source)
            if source in self.sender_counts:
                weighted_weight = self.weights_by_source[source] * self.sender_counts[source]
            else:
                weighted_weight = 0
            self.total_weight += weighted_weight
        #self.total_weight = self.total_weight*2 if self.total_weight < 0 else self.total_weight*.2
        self.total_weight = self.total_weight*2.9 if self.total_weight < 0 else self.total_weight
        return round(self.total_weight,2) 