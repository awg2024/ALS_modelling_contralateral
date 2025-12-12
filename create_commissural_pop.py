#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
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


class commissural_population:
    def __init__(self):
        # Separate stores for each commissural type
        self.v0d_tonic = []
        self.v0d_bursting = []
        self.v0v_tonic = []
        self.v0v_bursting = []
        self.v0c_tonic = []
        self.v0c_bursting = []

        self.v0d_all = []
        self.v0v_all = []
        self.v0c_all = []

        # monitors
        self.spike_detectors = []
        self.multimeters = []
        self.noise_generators = []

        self.spike_detector = None
        self.multimeter = None
        self.noise_generator = None

    def create_commissural_population(self, pop_type, self_connection, firing_behavior, pop_size, input_type):
        # --- Create common monitors ---
        spike_detector = nest.Create("spike_recorder", pop_size)
        multimeter = nest.Create("multimeter", netparams.mm_params)

        # --- Set input current and neuron params ---
        if firing_behavior == "tonic":

            # Determine I_e depending on input source
            if input_type == "descending":
                I_e = nest.random.normal(netparams.I_e_tonic_mean,
                                        netparams.I_e_tonic_std)

            elif input_type == "sensory_feedback":
                I_e = nest.random.normal(netparams.I_fb_tonic_mean,
                                        netparams.I_fb_tonic_std)

            else:
                I_e = 0 

            neuron_params = {
                'C_m': nest.random.normal(netparams.C_m_tonic_mean, netparams.C_m_tonic_std),
                'g_L': 10., 'E_L': -70.,
                'V_th': nest.random.normal(netparams.V_th_mean_tonic, netparams.V_th_std_tonic),
                'Delta_T': 2., 'tau_w': 30., 'a': 3., 'b': 0.,
                'V_reset': -58., 'I_e': I_e,
                't_ref': nest.random.normal(netparams.t_ref_mean, netparams.t_ref_std),
                'V_m': nest.random.normal(netparams.V_m_mean, netparams.V_m_std),
                "tau_syn_rise_I": netparams.tau_syn_i_rise,
                "tau_syn_decay_I": netparams.tau_syn_i_decay,
                "tau_syn_rise_E": netparams.tau_syn_e_rise,
                "tau_syn_decay_E": netparams.tau_syn_e_decay
            }
            noise_params = netparams.noise_params_tonic

         # --- Set input current and neuron params ---
        if firing_behavior == "bursting":

            # Determine I_e depending on input source
            if input_type == "descending":
                I_e = nest.random.normal(netparams.I_e_tonic_mean,
                                        netparams.I_e_tonic_std)

            elif input_type == "sensory_feedback":
                I_e = nest.random.normal(netparams.I_fb_tonic_mean,
                                        netparams.I_fb_tonic_std)

            else:
                I_e = 0 


            tau_rise_E = netparams.tau_syn_e_rise_rc if pop_type != "V0C" else netparams.tau_syn_e_rise
            tau_decay_E = netparams.tau_syn_e_decay_rc if pop_type != "V0C" else netparams.tau_syn_e_decay
            neuron_params = {
                'C_m': nest.random.normal(netparams.C_m_bursting_mean, netparams.C_m_bursting_std),
                'g_L': 26., 'E_L': -60.,
                'V_th': nest.random.normal(netparams.V_th_mean_bursting, netparams.V_th_std_bursting),
                'Delta_T': 2., 'tau_w': 130., 'a': -11., 'b': 30.,
                'V_reset': -48., 'I_e': I_e,
                't_ref': nest.random.normal(netparams.t_ref_mean, netparams.t_ref_std),
                'V_m': nest.random.normal(netparams.V_m_mean, netparams.V_m_std),
                "tau_syn_rise_I": netparams.tau_syn_i_rise,
                "tau_syn_decay_I": netparams.tau_syn_i_decay,
                "tau_syn_rise_E": tau_rise_E,
                "tau_syn_decay_E": tau_decay_E
            }
            noise_params = netparams.noise_params_bursting

        # --- Create neuron population ---
        neurons = nest.Create("aeif_cond_beta_aeif_cond_beta_nestml", pop_size, neuron_params)
        noise = nest.Create("noise_generator", noise_params)

        # --- Store population references ---
        if pop_type == "V0D":
            if firing_behavior == "tonic":
                self.v0d_tonic = neurons
            else:
                self.v0d_bursting = neurons
        elif pop_type == "V0V":
            if firing_behavior == "tonic":
                self.v0v_tonic = neurons
            else:
                self.v0v_bursting = neurons
        elif pop_type == "V0C":
            if firing_behavior == "tonic":
                self.v0c_tonic = neurons
            else:
                self.v0c_bursting = neurons

        # Maintain full population list as Python list of GIDs

        if pop_type == "V0D":
            self.v0d_all = list(self.v0d_tonic) + list(self.v0d_bursting)

        elif pop_type == "V0V":
            self.v0v_all = list(self.v0v_tonic) + list(self.v0v_bursting)

        elif pop_type == "V0C":
            self.v0c_all = list(self.v0c_tonic) + list(self.v0c_bursting)



        # --- Self-connections ---
        if self_connection in ["inh", "exc"]:
            syn_params = {
                "synapse_model": "static_synapse",
                "weight": nest.random.normal(
                    netparams.w_inh_mean if self_connection == "inh" else netparams.w_exc_mean,
                    netparams.w_inh_std if self_connection == "inh" else netparams.w_exc_std
                ),
                "delay": netparams.synaptic_delay
            }
            nest.Connect(neurons, neurons, "all_to_all", syn_params)

        # --- Connect noise and monitors ---
        nest.Connect(noise, neurons, "all_to_all")
        nest.Connect(neurons, spike_detector, "one_to_one")
        nest.Connect(multimeter, neurons)

        # --- Append to bookkeeping lists ---
        self.spike_detectors.append(spike_detector)
        self.spike_detector = spike_detector  # backward-compatible pointer

        self.multimeters.append(multimeter)
        self.multimeter = multimeter

        self.noise_generators.append(noise)
        self.noise_generator = noise

        return neurons