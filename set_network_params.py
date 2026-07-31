#!/usr/bin/env python

import nest
import numpy as np
import pathlib, sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import pickle, yaml
import time 
import datetime
import re
import os 
#nn=netparams.neural_network()

class neural_network():

    def __init__(self, config_path="configuration_run_nest.yaml", argv=None):

        # --- argv handling ---
        if argv is None:
            argv = sys.argv
        self.argv = argv

        # --- load yaml ---
        config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        self.args = args

        # --- handle days_after_onset from YAML ---
        days_from_yaml = args.get("days_after_onset", 0)
        if isinstance(days_from_yaml, list):
            self.days = [int(d) for d in days_from_yaml]
        else:
            self.days = [int(days_from_yaml)]  # wrap single value in list

        self.freq_test = int(args.get("freq_test", 1))
        self.bifurcation_test = args.get("bifurcation_test", "")
        
        
        source = "YAML"
        if len(argv) > 2:
            input_string = argv[2]
            m = re.fullmatch(r"P(\d+)_D(\d+)", input_string)
          
            if m:
                cli_days = int(m.group(1))
                cli_freq = int(m.group(2))

                self.freq_test = cli_freq

                if cli_days == 0:
                    self.days = 0
                    source = "CLI healthy override"
                elif cli_days > 0:
                    self.days = cli_days
                    source = "CLI disease override"


        # print(f"[CONFIG] days_after_onset = {self.days} ({source})")
        # print(f"[CONFIG] freq_test       = {self.freq_test}")

        self.save_all_pops = args['save_all_pops']
        self.asymmetric_onset = args['asymmetric_onset']
        self.save_results = args['save_results']

        self.asym_side = args.get('asym_side', 'L')   # side we are degenerating 
        self.bifurication_tag = args['bifurication_tag']
        
        stabilized = int(args['synaptically_stabilized'])
        compensation = int(args['synaptic_compensation'])

        if self.bifurcation_test == 0:
            run_tag = f"P{self.days}_D{self.freq_test}"  
        elif self.bifurcation_test == 1:
            run_tag = str(self.bifurication_tag)

        self.run_tag = run_tag 
        print("RUN TAG:", self.run_tag)


        # defining healthy dict for fall back in asymmetric degeneration 
        healthy_params = dict(
            v1_pct_surviving=1.0,  #Healthy=1., P63=.8084, P112=.5465 
            rc_pop_size=30, #Healthy=30, P63=14, P112=6
            v1a_pop_size=60,  #Healthy=60, P63=35, P112=33
            num_motor_neurons=150, #Healthy=150, P63=150, P112=107 (28% loss)
            v2a_tonic_pop_size=158, #Healthy=158, P63=158, P112=67 (59% loss)
            v2a_bursting_pop_size=0, # mammalian mouse found 50/50 of burst and tonic pop 
            w_1a_multiplier=1.0, # sorted - individual means and std for the connections 
            w_rc_multiplier=1.0, # sorted - individual means and std for the connections 
            w_v1_multiplier=1.0, # sorted - individual means and std for the connections 
            v0c_pop_size=15, #Healthy=15 Aligns with motor population of 10:1 Suggested by Garath Miles. 
            inh_inter_pop_size=300, #Healthy(V1/V2b)=300, use v1_pct_surviving to adjust cell death of V1 interneurons 
            v0d_pop_size = 75,  # Camerlo instruction 
            v0v_pop_size = 75, # Camerlo instruction 
            fb_multiplier=0.5, # issue. 
        )

        for side in ["L", "R"]:
            for k, v in healthy_params.items():
                setattr(self, f"{k}_{side}", v)
                #self.rc_pop_size_L/R


        
        # ==================== DEFINING DEFAULTS (Healthy Generic)==================
        # sparsity 
        self.sparsity_rg_v2b = 0.5 
        self.sparsity_v2b_rg = 0.5 
        self.sparsity_rg_v1 = 0.5 
        self.sparsity_v1_rg = 0.5
        self.sparsity_v1s_outside_rg_layer = 0.28 
        self.sparsity_custom_v0d_rg_inh = 0.15     
        self.current_multiplier_bursting_flx = 1.5 # Use this to change network output frequency
        self.current_multiplier_tonic_flx = 1.5 
        self.current_multiplier_bursting_ext = 2.8 #Use this to change ext MN firing rate, increased input = increased firing 
        self.current_multiplier_tonic_ext = 2.8      
        self.v2b_current_multiplier = 0.38 #Use this to reduce initial bump in RG ext output 
        self.v1_current_multiplier = 0.3 #Use this to change ext MNP BD, increased input = increased width  
        self.inh_weight_multiplier = 1  
        self.exc_weight_multiplier = 2

        # sparsity further params controlled by two variables 
        self.sparsity_custom_v1_rg = self.sparsity_v1_rg   
        self.sparsity_custom_v1_v2a = self.sparsity_v1s_outside_rg_layer 
        self.sparsity_custom_v1_mn = self.sparsity_v1s_outside_rg_layer 
        self.sparsity_custom_rc_mn = self.sparsity_v1s_outside_rg_layer 
        self.sparsity_custom_v1a_mn = self.sparsity_v1s_outside_rg_layer
        self.sparsity_custom_rc_v1a = self.sparsity_v1s_outside_rg_layer 
        self.sparsity_custom_rc_rc = self.sparsity_v1s_outside_rg_layer
        self.sparsity_custom_v1a_v1a = self.sparsity_v1s_outside_rg_layer  
            
        # RG self connectivity params 
        self.selfexc_flx = 0.45  #Increase self-connectivity to produce oscillation with less neurons TESTING 0.5
        self.selfexc_ext = 0.15 # testing 0.2
  

        self.conn_dict_custom_selfexc_flx = {'rule': 'pairwise_bernoulli', 'p': self.selfexc_flx}
        self.conn_dict_custom_selfexc_ext = {'rule': 'pairwise_bernoulli', 'p': self.selfexc_ext}

        #V1 synaptic strength
        self.w_1a_multiplier = 1  
        self.w_rc_multiplier = 1 
        self.w_v1_multiplier = 1 

         # Descending V0d inh drive- Healthy -180, Silenced -450 for silencing experiments 
        self.I_e_tonic_inh_R_mean = -180
        self.I_e_tonic_inh_L_mean = -180 # baseline -180 for both 

          # V3 exc drive pA
        self.I_e_tonic_v3_L_mean = 180 # baseline pA is 180 for Spiking V0d ? We can just apply the inverse at a low connectivity and drives
        self.I_e_tonic_v3_R_mean = 180 # baseline 180 for both 

          # danner 2017 = -0.07 = 0.24 for the v3 translate it from a baseline we already know
        # V0d contrlateral connectivity params - 
        self.w_custom_v0d_rg_inh_R_mean = -0.58 # v0d inhibiting rg. Defined and Tested. 
        self.w_custom_v0d_rg_inh_R_std = 0.058
        self.w_custom_v0d_rg_inh_L_mean = -0.58 
        self.w_custom_v0d_rg_inh_L_std = 0.058
        self.w_custom_v0d_rg_inh_mean = -0.58 
        self.w_custom_v0d_rg_inh_std = 0.058


        # V3 Flx Flx connectivity mean and std 
        self.w_custom_v3_rg_L_mean = 0.837 # contralateral. weights tested and validated. 
        self.w_custom_v3_rg_L_std = 0.0837
        self.w_custom_v3_rg_R_mean = 0.837
        self.w_custom_v3_rg_R_std = 0.0837
        self.w_custom_v3_rg_mean = 0.837
        self.w_custom_v3_rg_std = 0.0837

        self.w_custom_rg_v3_L_mean = 0.29 *self.exc_weight_multiplier # ipsilateral v3 connection stregnths
        self.w_custom_rg_v3_L_std = 0.02 *self.exc_weight_multiplier
        self.w_custom_rg_v3_R_mean = 0.29 *self.exc_weight_multiplier
        self.w_custom_rg_v3_R_std = 0.02 *self.exc_weight_multiplier
       
        # population sizes 
        self.v1_pct_surviving=1.0  #Healthy=1., P63=.8084, P112=.5465 
        self.rc_pop_size=30 #Healthy=30, P63=14, P112=6
        self.v1a_pop_size=60  #Healthy=60, P63=35, P112=33
        self.num_motor_neurons=150 #Healthy=150, P63=150, P112=107 (28% loss)
        self.v2a_tonic_pop_size=158 #Healthy=158, P63=158, P112=67 (59% loss)
        self.v2a_bursting_pop_size=0 # mammalian mouse found 50/50 of burst and tonic pop 
        self.w_1a_multiplier=1.0 # sorted - individual means and std for the connections 
        self.w_rc_multiplier=1.0 # sorted - individual means and std for the connections 
        self.w_v1_multiplier=1.0 # sorted - individual means and std for the connections 
        self.v0c_pop_size=15 #Healthy=15 Aligns with motor population of 10:1 Suggested by Garath Miles. 
        self.inh_inter_pop_size=300 #Healthy(V1/V2b)=300, use v1_pct_surviving to adjust cell death of V1 interneurons 
        self.v0d_pop_size = 75  # Camerlo instruction 
        self.v0v_pop_size = 75 # Camerlo instruction 
        self.fb_multiplier=0.5
        self.v3_pop_size = 100 # TESTING. 

        # ================== DAY 0 P0_D0 ==============================
        if self.days == 0:

            print("[INFO] Running simulation P0 (healthy)")

            # Not bothered about asymmetric or symmetric degeneration as this is a healthy condition 
            if self.asymmetric_onset == 1 or self.asymmetric_onset == 0: 

                print("[INFO] Asymmetric or Symmetric Parameters have no influence as there is no degeneration.")

                for side in ("L", "R"): 
                        
                    # sparsity params  
                    setattr(self, f"sparsity_custom_v1_rg_{side}", self.sparsity_v1_rg)
                    setattr(self, f"sparsity_custom_v1_v2a_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_v1_mn_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_rc_mn_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_v1a_mn_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_rc_v1a_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_rc_rc_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer)
                    setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", self.sparsity_custom_v0d_rg_inh) # v0d contra connectivity implemented
                
                    # multipliers
                    setattr(self, f"w_1a_multiplier_{side}", healthy_params["w_1a_multiplier"])
                    setattr(self, f"w_rc_multiplier_{side}", healthy_params["w_rc_multiplier"])
                    setattr(self, f"w_v1_multiplier_{side}", healthy_params["w_v1_multiplier"])
                    setattr(self, f"fb_multiplier_{side}",   healthy_params["fb_multiplier"])

                    # pops / survival
                    setattr(self, f"inh_inter_pop_size_{side}",    healthy_params["inh_inter_pop_size"])
                    setattr(self, f"v1_pct_surviving_{side}",      healthy_params["v1_pct_surviving"])
                    setattr(self, f"v2a_tonic_pop_size_{side}",    healthy_params["v2a_tonic_pop_size"])
                    setattr(self, f"v2a_bursting_pop_size_{side}", healthy_params["v2a_bursting_pop_size"])
                    setattr(self, f"v1a_pop_size_{side}",          healthy_params["v1a_pop_size"])
                    setattr(self, f"v0c_pop_size_{side}",          healthy_params["v0c_pop_size"])
                    setattr(self, f"rc_pop_size_{side}",           healthy_params["rc_pop_size"])
                    setattr(self, f"num_motor_neurons_{side}",     healthy_params["num_motor_neurons"])
                    setattr(self, f"v0d_pop_size_{side}",          healthy_params["v0d_pop_size"])
                    setattr(self, f"v0v_pop_size_{side}",          healthy_params["v0v_pop_size"])

    
        # ======================== DAY 45 ==========================
        elif self.days == 45:
            
            print("[INFO] Running simulation P45")

            # Start by defining HEALTHY parameters for both sides. Keeping Defaults. 
            for side in ("L", "R"):
                    
                # multipliers
                setattr(self, f"w_1a_multiplier_{side}", healthy_params["w_1a_multiplier"])
                setattr(self, f"w_rc_multiplier_{side}", healthy_params["w_rc_multiplier"])
                setattr(self, f"w_v1_multiplier_{side}", healthy_params["w_v1_multiplier"])
                setattr(self, f"fb_multiplier_{side}",   healthy_params["fb_multiplier"])

                # pops / survival
                setattr(self, f"inh_inter_pop_size_{side}",    healthy_params["inh_inter_pop_size"])
                setattr(self, f"v1_pct_surviving_{side}",      healthy_params["v1_pct_surviving"])
                setattr(self, f"v2a_tonic_pop_size_{side}",    healthy_params["v2a_tonic_pop_size"])
                setattr(self, f"v2a_bursting_pop_size_{side}", healthy_params["v2a_bursting_pop_size"])
                setattr(self, f"v1a_pop_size_{side}",          healthy_params["v1a_pop_size"])
                setattr(self, f"v0c_pop_size_{side}",          healthy_params["v0c_pop_size"])
                setattr(self, f"rc_pop_size_{side}",           healthy_params["rc_pop_size"])
                setattr(self, f"num_motor_neurons_{side}",     healthy_params["num_motor_neurons"])
                setattr(self, f"v0d_pop_size_{side}",          healthy_params["v0d_pop_size"])
                setattr(self, f"v0v_pop_size_{side}",          healthy_params["v0v_pop_size"])

                # sparsity params  
                setattr(self, f"sparsity_custom_v1_rg_{side}", (self.sparsity_v1_rg))
                setattr(self, f"sparsity_custom_v1_v2a_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_v1_mn_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_rc_mn_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_v1a_mn_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_rc_v1a_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_rc_rc_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_v1a_v1a_{side}", (self.sparsity_v1s_outside_rg_layer))
                setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", self.sparsity_custom_v0d_rg_inh) # v0d contra connectivity implemented
                

            if self.asymmetric_onset == 1:
                
                print("[INFO] Running Asymmetric Degeneration P45")

                # Decide affected vs healthy side labels
                if self.asym_side == "L":
                    A, H = "L", "R"
                else: # Else asym_side = R... 
                    A, H = "R", "L"
                # A = affected & H = Healthy 

                # feedback
                setattr(self, f"fb_multiplier_{A}", 0.5)

                # sparsity (halve)
                setattr(self, f"sparsity_custom_v1_rg_{A}", self.sparsity_v1_rg / 2)
                setattr(self, f"sparsity_custom_v1_v2a_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1_mn_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_mn_{A}",  self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1a_mn_{A}",  self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_v1a_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_rc_{A}", self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1a_v1a_{A}", self.sparsity_v1s_outside_rg_layer / 2)
     
        
                # synaptic strength (compensation only on affected side)
                setattr(self, f"w_1a_multiplier_{A}", 2.5 if compensation == 1 else 1.0)
                setattr(self, f"w_rc_multiplier_{A}", 2.5 if compensation == 1 else 1.0)
                setattr(self, f"w_v1_multiplier_{A}", 2.0 if compensation == 1 else 1.0)

                # population sizes (affected)
                setattr(self, f"inh_inter_pop_size_{A}",    300)
                setattr(self, f"v1_pct_surviving_{A}",      1.0)
                setattr(self, f"v2a_tonic_pop_size_{A}",    158)
                setattr(self, f"v2a_bursting_pop_size_{A}", 0)
                setattr(self, f"v1a_pop_size_{A}",          60)
                setattr(self, f"v0c_pop_size_{A}",          15)
                setattr(self, f"rc_pop_size_{A}",           30)
                setattr(self, f"num_motor_neurons_{A}",     150)
         
                if args['v0d_commissural_degeneration'] == 1: 

                    attr_name = f"w_custom_v0d_rg_inh_{A}_mean"
                    baseline_v0d_strength = getattr(self, attr_name)
                    setattr(self, f"sparsity_custom_v0d_rg_inh_{A}", 0.12)  # V0d sparsity baseline 0.1275, reducing the sparsity reduced by 2.5%
                    
                if args['v3_commissural_hyperexcitation'] == 1: 

                    attr_name = f"w_custom_v3_rg_{A}_mean"
                    baseline_v3_strength = getattr(self, attr_name)
                    setattr(self, attr_name, baseline_v3_strength * 1.05)  # 10% increase in synaptic strength from V3
            

            elif self.asymmetric_onset == 0: 
                
                print("[INFO] Running Symmetric Degeneration (both Hemicords affected) P45 ")

                for side in ("L", "R"):

                    setattr(self, f"fb_multiplier_{side}", 0.5)

                     # sparsity (halve)
                    setattr(self, f"sparsity_custom_v1_rg_{side}", (self.sparsity_v1_rg)/2)
                    setattr(self, f"sparsity_custom_v1_v2a_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                    setattr(self, f"sparsity_custom_v1_mn_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                    setattr(self, f"sparsity_custom_rc_mn_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                    setattr(self, f"sparsity_custom_v1a_mn_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                    setattr(self, f"sparsity_custom_rc_v1a_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                    setattr(self, f"sparsity_custom_rc_rc_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                    setattr(self, f"sparsity_custom_v1a_v1a_{side}", (self.sparsity_v1s_outside_rg_layer)/2)
                
                    setattr(self, f"w_1a_multiplier_{side}", 2.5 if compensation == 1 else 1.0)
                    setattr(self, f"w_rc_multiplier_{side}", 2.5 if compensation == 1 else 1.0)
                    setattr(self, f"w_v1_multiplier_{side}", 2.0 if compensation == 1 else 1.0)

                    # population sizes (affected)
                    setattr(self, f"inh_inter_pop_size_{side}",    300)
                    setattr(self, f"v1_pct_surviving_{side}",      1.0)
                    setattr(self, f"v2a_tonic_pop_size_{side}",    158)
                    setattr(self, f"v2a_bursting_pop_size_{side}", 0)
                    setattr(self, f"v1a_pop_size_{side}",          60)
                    setattr(self, f"v0c_pop_size_{side}",          15)
                    setattr(self, f"rc_pop_size_{side}",           30)
                    setattr(self, f"num_motor_neurons_{side}",     150)
                    setattr(self, f"v0d_pop_size_{side}", 75)                  

        
                    if args['v0d_commissural_degeneration'] == 1: 

                        attr_name = f"w_custom_v0d_rg_inh_{side}_mean"
                        baseline_v0d_strength = getattr(self, attr_name)
                        setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", 0.12)  # V0d sparsity baseline 0.1275, reducing the sparsity reduced by 2.5%
                    
                    if args['v3_commissural_hyperexcitation'] == 1: 

                        attr_name = f"w_custom_v3_rg_{side}_mean"
                        baseline_v3_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v3_strength * 1.05)  # 10% increase in synaptic strength from V3
            
                

        # ============================= END OF DAY 45 ====================================


        # ============================= DAY 63 ====================================
        elif self.days == 63:
           
            print("[INFO] Running simulation P63")


            # Start by defining HEALTHY parameters for both sides. Keeping Defaults. 
            for side in ("L", "R"):
                    
                # multipliers
                setattr(self, f"w_1a_multiplier_{side}", healthy_params["w_1a_multiplier"])
                setattr(self, f"w_rc_multiplier_{side}", healthy_params["w_rc_multiplier"])
                setattr(self, f"w_v1_multiplier_{side}", healthy_params["w_v1_multiplier"])
                setattr(self, f"fb_multiplier_{side}",   healthy_params["fb_multiplier"])

                # pops / survival
                setattr(self, f"inh_inter_pop_size_{side}",    healthy_params["inh_inter_pop_size"])
                setattr(self, f"v1_pct_surviving_{side}",      healthy_params["v1_pct_surviving"])
                setattr(self, f"v2a_tonic_pop_size_{side}",    healthy_params["v2a_tonic_pop_size"])
                setattr(self, f"v2a_bursting_pop_size_{side}", healthy_params["v2a_bursting_pop_size"])
                setattr(self, f"v1a_pop_size_{side}",          healthy_params["v1a_pop_size"])
                setattr(self, f"v0c_pop_size_{side}",          healthy_params["v0c_pop_size"])
                setattr(self, f"rc_pop_size_{side}",           healthy_params["rc_pop_size"])
                setattr(self, f"num_motor_neurons_{side}",     healthy_params["num_motor_neurons"])
                setattr(self, f"v0d_pop_size_{side}",          healthy_params["v0d_pop_size"])
                setattr(self, f"v0v_pop_size_{side}",          healthy_params["v0v_pop_size"])
                 

                # sparsity params  
                setattr(self, f"sparsity_custom_v1_rg_{side}",   self.sparsity_v1_rg)
                setattr(self, f"sparsity_custom_v1_v2a_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1_mn_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_mn_{side}",  self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1a_mn_{side}",  self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_v1a_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_rc_{side}", self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", self.sparsity_custom_v0d_rg_inh) # v0d contra connectivity implemented
                
                

            if self.asymmetric_onset == 1:

                print("[INFO] Running Asymmetric Degeneration P63")

                # Decide affected vs healthy side labels
                if self.asym_side == "L":
                    A, H = "L", "R"
                else: # Else asym_side = R... 
                    A, H = "R", "L"
                # A = affected & H = Healthy 

            
                # feedback
                setattr(self, f"fb_multiplier_{A}", 0.5)

                # sparsity (halve)
                setattr(self, f"sparsity_custom_v1_rg_{A}",   self.sparsity_custom_v1_rg/2)
                setattr(self, f"sparsity_custom_v1_v2a_{A}",   self.sparsity_v1s_outside_rg_layer/2)
                setattr(self, f"sparsity_custom_v1_mn_{A}",   self.sparsity_v1s_outside_rg_layer/2)
                setattr(self, f"sparsity_custom_rc_mn_{A}",  self.sparsity_v1s_outside_rg_layer/2)
                setattr(self, f"sparsity_custom_v1a_mn_{A}",  self.sparsity_v1s_outside_rg_layer/2)
                setattr(self, f"sparsity_custom_rc_v1a_{A}",   self.sparsity_v1s_outside_rg_layer/2)
                setattr(self, f"sparsity_custom_rc_rc_{A}", self.sparsity_v1s_outside_rg_layer/2)
                setattr(self, f"sparsity_custom_v1a_v1a_{A}", self.sparsity_v1s_outside_rg_layer/2)
           
            
                # synaptic strength (compensation only on affected side)
                setattr(self, f"w_1a_multiplier_{A}", 4.0 if compensation == 1 else 1.0)
                setattr(self, f"w_rc_multiplier_{A}", 4.0 if compensation == 1 else 1.0)
                setattr(self, f"w_v1_multiplier_{A}", 3.0 if compensation == 1 else 1.0)

                # population sizes (affected 63)
                setattr(self, f"inh_inter_pop_size_{A}",    300)
                setattr(self, f"v1_pct_surviving_{A}",      0.8084)
                setattr(self, f"v2a_tonic_pop_size_{A}",    158)
                setattr(self, f"v2a_bursting_pop_size_{A}", 0)
                setattr(self, f"v1a_pop_size_{A}",          35)
                setattr(self, f"v0c_pop_size_{A}",          15)
                setattr(self, f"rc_pop_size_{A}",           14)
                setattr(self, f"num_motor_neurons_{A}",     150)

                if args['v0d_commissural_degeneration'] == 1: 

                    attr_name = f"w_custom_v0d_rg_inh_{A}_mean"
                    baseline_v0d_strength = getattr(self, attr_name)
                    setattr(self, attr_name, baseline_v0d_strength * 0.975)  # 5% loss in synaptic strength from V0d 
                    setattr(self, f"sparsity_custom_v0d_rg_inh_{A}", 0.11475)  # V0d sparsity baseline 0.1275, reducing the sparsity reduced by 10%
                    setattr(self, f"v0d_pop_size_{A}", 60)                  # V0d pop size baseline 64, further reducing inhibitory pool

                if args['v3_commissural_hyperexcitation'] == 1: 

                    attr_name = f"w_custom_v3_rg_{A}_mean"
                    baseline_v3_strength = getattr(self, attr_name)
                    setattr(self, attr_name, baseline_v3_strength * 1.10)  # 10% increase in synaptic strength from V3
          
                
            elif self.asymmetric_onset == 0:

                print("[INFO] Running Symmetric Degeneration (both hemicords affected) P63 ")

                for side in ("L", "R"):

                    setattr(self, f"fb_multiplier_{side}", 0.5)
                                    
                    # sparsity (halve)
                    setattr(self, f"sparsity_custom_v1_rg_{side}",   self.sparsity_custom_v1_rg/2)
                    setattr(self, f"sparsity_custom_v1_v2a_{side}",   self.sparsity_v1s_outside_rg_layer/2)
                    setattr(self, f"sparsity_custom_v1_mn_{side}",   self.sparsity_v1s_outside_rg_layer/2)
                    setattr(self, f"sparsity_custom_rc_mn_{side}",  self.sparsity_v1s_outside_rg_layer/2)
                    setattr(self, f"sparsity_custom_v1a_mn_{side}",  self.sparsity_v1s_outside_rg_layer/2)
                    setattr(self, f"sparsity_custom_rc_v1a_{side}",   self.sparsity_v1s_outside_rg_layer/2)
                    setattr(self, f"sparsity_custom_rc_rc_{side}", self.sparsity_v1s_outside_rg_layer/2)
                    setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer/2)
             
                    setattr(self, f"w_1a_multiplier_{side}", 4.0 if compensation == 1 else 1.0)
                    setattr(self, f"w_rc_multiplier_{side}", 4.0 if compensation == 1 else 1.0)
                    setattr(self, f"w_v1_multiplier_{side}", 3.0 if compensation == 1 else 1.0)

                    # population sizes (affected)
                    setattr(self, f"inh_inter_pop_size_{side}",    300)
                    setattr(self, f"v1_pct_surviving_{side}",      0.8084)
                    setattr(self, f"v2a_tonic_pop_size_{side}",    158)
                    setattr(self, f"v2a_bursting_pop_size_{side}", 0)
                    setattr(self, f"v1a_pop_size_{side}",          35)
                    setattr(self, f"v0c_pop_size_{side}",          15)
                    setattr(self, f"rc_pop_size_{side}",           14)
                    setattr(self, f"num_motor_neurons_{side}",     150)
                
                    if args['v0d_commissural_degeneration'] == 1: 

                        attr_name = f"w_custom_v0d_rg_inh_{side}_mean"
                        baseline_v0d_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v0d_strength * 0.975)  # 5% loss in synaptic strength from V0d 
                        setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", 0.11475)  # V0d sparsity baseline 0.1275, reducing the sparsity reduced by 10%
                        setattr(self, f"v0d_pop_size_{side}", 60)                  # V0d pop size baseline 64, further reducing inhibitory pool

                    if args['v3_commissural_hyperexcitation'] == 1: 

                        attr_name = f"w_custom_v3_rg_{side}_mean"
                        baseline_v3_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v3_strength * 1.10)  # 10% increase in synaptic strength from V3

                        


                
                 # ========================= END OF P63 =========================

        
        # ========================= START OF P112 STABILISED 0 =============================
        elif self.days == 112 and stabilized == 0:

            print("Running Simulation P112")

            # Start by defining HEALTHY parameters for both sides. Keeping Defaults. 
            for side in ("L", "R"):
                        
                # multipliers
                setattr(self, f"w_1a_multiplier_{side}", healthy_params["w_1a_multiplier"])
                setattr(self, f"w_rc_multiplier_{side}", healthy_params["w_rc_multiplier"])
                setattr(self, f"w_v1_multiplier_{side}", healthy_params["w_v1_multiplier"])
                setattr(self, f"fb_multiplier_{side}",   healthy_params["fb_multiplier"])

                # pops / survival
                setattr(self, f"inh_inter_pop_size_{side}",    healthy_params["inh_inter_pop_size"])
                setattr(self, f"v1_pct_surviving_{side}",      healthy_params["v1_pct_surviving"])
                setattr(self, f"v2a_tonic_pop_size_{side}",    healthy_params["v2a_tonic_pop_size"])
                setattr(self, f"v2a_bursting_pop_size_{side}", healthy_params["v2a_bursting_pop_size"])
                setattr(self, f"v1a_pop_size_{side}",          healthy_params["v1a_pop_size"])
                setattr(self, f"v0c_pop_size_{side}",          healthy_params["v0c_pop_size"])
                setattr(self, f"rc_pop_size_{side}",           healthy_params["rc_pop_size"])
                setattr(self, f"num_motor_neurons_{side}",     healthy_params["num_motor_neurons"])
                setattr(self, f"v0d_pop_size_{side}",          healthy_params["v0d_pop_size"])
                setattr(self, f"v0v_pop_size_{side}",          healthy_params["v0v_pop_size"])

                # sparsity params  
                setattr(self, f"sparsity_custom_v1_rg_{side}",   self.sparsity_custom_v1_rg)
                setattr(self, f"sparsity_custom_v1_v2a_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1_mn_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_mn_{side}",  self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1a_mn_{side}",  self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_v1a_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_rc_{side}", self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", self.sparsity_custom_v0d_rg_inh) # v0d contra connectivity implemented
                

            if self.asymmetric_onset == 1:
                
                print("[INFO] Running Asymmetric Degeneration")

             # Decide affected vs healthy side labels
                if self.asym_side == "L":
                    A, H = "L", "R"
                else: # Else asym_side = R... 
                    A, H = "R", "L"
                 # A = affected & H = Healthy 

                # feedback
                setattr(self, f"fb_multiplier_{A}", 0.5)

                # sparsity (halve)
                setattr(self, f"sparsity_custom_v1_rg_{A}",   self.sparsity_v1_rg / 2)
                setattr(self, f"sparsity_custom_v1_v2a_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1_mn_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_mn_{A}",  self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1a_mn_{A}",  self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_v1a_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_rc_{A}", self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1a_v1a_{A}", self.sparsity_v1s_outside_rg_layer / 2)
                
                # synaptic strength (compensation only on affected side)
                setattr(self, f"w_1a_multiplier_{A}", 4.5 if compensation == 1 else 1.0)
                setattr(self, f"w_rc_multiplier_{A}", 4.5 if compensation == 1 else 1.0)
                setattr(self, f"w_v1_multiplier_{A}", 4.0 if compensation == 1 else 1.0)
                setattr(self, f"w_v2a_multiplier_{A}", 2.0)

                # population sizes (affected 63)
                setattr(self, f"inh_inter_pop_size_{A}",    300)
                setattr(self, f"v1_pct_surviving_{A}",      0.5465)
                
                if args['v2a_intact'] == 0:
                    setattr(self, f"v2a_tonic_pop_size_{A}",    67)
                elif args['v2a_intact'] == 1:
                    setattr(self, f"v2a_tonic_pop_size_{A}",    158)

                setattr(self, f"v2a_bursting_pop_size_{A}", 0)
                setattr(self, f"v1a_pop_size_{A}",          33)
                setattr(self, f"v0c_pop_size_{A}",          15)
                setattr(self, f"rc_pop_size_{A}",           6)
                
                if args['mn_intact'] == 0:
                    setattr(self, f"num_motor_neurons_{A}",    107)
                elif args['mn_intact'] == 1:
                    setattr(self, f"num_motor_neurons_{A}",    150)

                if args['v0d_commissural_degeneration'] == 1: 

                    attr_name = f"w_custom_v0d_rg_inh_{A}_mean"
                    baseline_v0d_strength = getattr(self, attr_name)
                    setattr(self, attr_name, baseline_v0d_strength * 0.95)  # 5% loss in synaptic strength from V0d 
                    setattr(self, f"sparsity_custom_v0d_rg_inh_{A}", 0.08)  # V0d sparsity baseline 0.1275, reducing the sparsity
                    setattr(self, f"v0d_pop_size_{A}", 48)                  # V0d pop size baseline 64, further reducing inhibitory pool

                if args['v3_commissural_hyperexcitation'] == 1: 

                    attr_name = f"w_custom_v3_rg_{A}_mean"
                    baseline_v3_strength = getattr(self, attr_name)
                    setattr(self, attr_name, baseline_v3_strength * 1.25)  # 25% increase in synaptic strength from V3

                    
                
            elif self.asymmetric_onset == 0:

                print("[INFO] Running Symmetric Degeneration - Both Hemicords Affected ")

                for side in ("L", "R"):

                    setattr(self, f"fb_multiplier_{side}", 0.5)   # sparsity (halve)       
                    setattr(self, f"sparsity_custom_v1_rg_{side}",   self.sparsity_v1_rg / 2)
                    setattr(self, f"sparsity_custom_v1_v2a_{side}",   self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_v1_mn_{side}",   self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_rc_mn_{side}",   self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_v1a_mn_{side}",  self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_rc_v1a_{side}",  self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_rc_rc_{side}",   self.sparsity_v1s_outside_rg_layer/ 2)
                    setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer / 2)
                    
                  
                    setattr(self, f"w_1a_multiplier_{side}", 4.5 if compensation == 1 else 1.0)
                    setattr(self, f"w_rc_multiplier_{side}", 4.5 if compensation == 1 else 1.0)
                    setattr(self, f"w_v1_multiplier_{side}", 4.0 if compensation == 1 else 1.0)
                    setattr(self, f"w_v2a_multiplier_{side}", 2.0)

                    # population sizes (affected 112)
                    setattr(self, f"inh_inter_pop_size_{side}",    300)
                    setattr(self, f"v1_pct_surviving_{side}",      0.5465)
                    
                    if args['v2a_intact'] == 0:
                        setattr(self, f"v2a_tonic_pop_size_{side}",    67)
                    elif args['v2a_intact'] == 1:
                        setattr(self, f"v2a_tonic_pop_size_{side}",    158)

                    setattr(self, f"v2a_bursting_pop_size_{side}", 0)
                    setattr(self, f"v1a_pop_size_{side}",          33)
                    setattr(self, f"v0c_pop_size_{side}",          15)
                    setattr(self, f"rc_pop_size_{side}",           6)
                    
                    if args['mn_intact'] == 0:
                        setattr(self, f"num_motor_neurons_{side}",    107)
                    elif args['mn_intact'] == 1:
                        setattr(self, f"num_motor_neurons_{side}",    150)

                    if args['v0d_commissural_degeneration'] == 1: 

                        attr_name = f"w_custom_v0d_rg_inh_{side}_mean"
                        baseline_v0d_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v0d_strength * 0.95)  # 5% loss in synaptic strength from V0d 
                        setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", 0.08)  # V0d sparsity baseline 0.1275, reducing the sparsity
                        setattr(self, f"v0d_pop_size_{side}", 48)                  # V0d pop size baseline 64, further reducing inhibitory pool

                    if args['v3_commissural_hyperexcitation'] == 1: 

                        attr_name = f"w_custom_v3_rg_{side}_mean"
                        baseline_v3_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v3_strength * 1.25)  # 25% increase in synaptic strength from V3

        # ===================  DAY 112 STABILISED 1 ======================

        elif self.days==112 and stabilized==1:

            print('Running simulation P112 with synaptic stabilization')

            # Start by defining HEALTHY parameters for both sides. Keeping Defaults. 
            for side in ("L", "R"):
                            
                    # multipliers
                setattr(self, f"w_1a_multiplier_{side}", healthy_params["w_1a_multiplier"])
                setattr(self, f"w_rc_multiplier_{side}", healthy_params["w_rc_multiplier"])
                setattr(self, f"w_v1_multiplier_{side}", healthy_params["w_v1_multiplier"])
                setattr(self, f"fb_multiplier_{side}",   healthy_params["fb_multiplier"])

                    # pops / survival
                setattr(self, f"inh_inter_pop_size_{side}",    healthy_params["inh_inter_pop_size"])
                setattr(self, f"v1_pct_surviving_{side}",      healthy_params["v1_pct_surviving"])
                setattr(self, f"v2a_tonic_pop_size_{side}",    healthy_params["v2a_tonic_pop_size"])
                setattr(self, f"v2a_bursting_pop_size_{side}", healthy_params["v2a_bursting_pop_size"])
                setattr(self, f"v1a_pop_size_{side}",          healthy_params["v1a_pop_size"])
                setattr(self, f"v0c_pop_size_{side}",          healthy_params["v0c_pop_size"])
                setattr(self, f"rc_pop_size_{side}",           healthy_params["rc_pop_size"])
                setattr(self, f"num_motor_neurons_{side}",     healthy_params["num_motor_neurons"])
                setattr(self, f"v0d_pop_size_{side}",          healthy_params["v0d_pop_size"])
                setattr(self, f"v0v_pop_size_{side}",          healthy_params["v0v_pop_size"])

                    # sparsity params  
                setattr(self, f"sparsity_custom_v1_rg_{side}",   self.sparsity_custom_v1_rg)
                setattr(self, f"sparsity_custom_v1_v2a_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1_mn_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_mn_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1a_mn_{side}",  self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_v1a_{side}",  self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_rc_rc_{side}",   self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer)
                setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", self.sparsity_custom_v0d_rg_inh) # v0d contra connectivity implemented
                


            if self.asymmetric_onset == 1:
                    
                print("[INFO] Running Asymmetric Degeneration")

                # Decide affected vs healthy side labels
                if self.asym_side == "L":
                    A, H = "L", "R"
                else: # Else asym_side = R... 
                    A, H = "R", "L"
                # A = affected & H = Healthy 

                # feedback
                setattr(self, f"fb_multiplier_{A}", 0.5)

                # sparsity (halve)
                setattr(self, f"sparsity_custom_v1_rg_{A}",   self.sparsity_v1_rg / 2)
                setattr(self, f"sparsity_custom_v1_v2a_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1_mn_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_mn_{A}",  self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1a_mn_{A}",  self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_v1a_{A}",   self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_rc_rc_{A}", self.sparsity_v1s_outside_rg_layer / 2)
                setattr(self, f"sparsity_custom_v1a_v1a_{A}", self.sparsity_v1s_outside_rg_layer / 2)
        
                # synaptic strength (compensation only on affected side)
                setattr(self, f"w_1a_multiplier_{A}", 4.5 if compensation == 1 else 1.0)
                setattr(self, f"w_rc_multiplier_{A}", 4.5 if compensation == 1 else 1.0)
                setattr(self, f"w_v1_multiplier_{A}", 4.0 if compensation == 1 else 1.0)
                setattr(self, f"w_v2a_multiplier_{A}", 2.0)

                # population sizes (affected 63)
                setattr(self, f"inh_inter_pop_size_{A}",    300)
                setattr(self, f"v1_pct_surviving_{A}",      0.5465)
                    
                if args['v2a_intact'] == 0:
                    setattr(self, f"v2a_tonic_pop_size_{A}",    67)
                elif args['v2a_intact'] == 1:
                    setattr(self, f"v2a_tonic_pop_size_{A}",    158)

                setattr(self, f"v2a_bursting_pop_size_{A}", 0)
                setattr(self, f"v1a_pop_size_{A}",          33)
                setattr(self, f"v0c_pop_size_{A}",          15)
                setattr(self, f"rc_pop_size_{A}",           6)
            
                if args['mn_intact'] == 0:
                    setattr(self, f"num_motor_neurons_{A}",    107)
                elif args['mn_intact'] == 1:
                    setattr(self, f"num_motor_neurons_{A}",    150)

                if args['v0d_commissural_degeneration'] == 1: 

                        attr_name = f"w_custom_v0d_rg_inh_{A}_mean"
                        baseline_v0d_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v0d_strength * 0.95)  # 5% loss in synaptic strength from V0d 
                        setattr(self, f"sparsity_custom_v0d_rg_inh_{A}", 0.08)  # V0d sparsity baseline 0.1275, reducing the sparsity
                        setattr(self, f"v0d_pop_size_{A}", 48)                  # V0d pop size baseline 64, further reducing inhibitory pool

                if args['v3_commissural_hyperexcitation'] == 1: 

                        attr_name = f"w_custom_v3_rg_{A}_mean"
                        baseline_v3_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v3_strength * 1.25)  # 25% increase in synaptic strength from V3


                    
            elif self.asymmetric_onset == 0:

                print("[INFO] Running Symmetric Degeneration - Both Hemicords Affected ")

                for side in ("L", "R"):

                    setattr(self, f"fb_multiplier_{side}", 0.5)

                                            # sparsity (halve)
                    setattr(self, f"sparsity_custom_v1_rg_{side}",   self.sparsity_v1_rg / 2)
                    setattr(self, f"sparsity_custom_v1_mn_{side}",   self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_rc_mn_{side}",   self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_v1a_mn_{side}",  self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_rc_v1a_{side}",  self.sparsity_v1s_outside_rg_layer / 2)
                    setattr(self, f"sparsity_custom_rc_rc_{side}",   self.sparsity_v1s_outside_rg_layer/ 2)
                    setattr(self, f"sparsity_custom_v1a_v1a_{side}", self.sparsity_v1s_outside_rg_layer / 2)
               
                    setattr(self, f"w_1a_multiplier_{side}", 2.5 if compensation == 1 else 1.0)
                    setattr(self, f"w_rc_multiplier_{side}", 2.5 if compensation == 1 else 1.0)
                    setattr(self, f"w_v1_multiplier_{side}", 2.0 if compensation == 1 else 1.0)
                    setattr(self, f"w_v2a_multiplier_{side}", 2.0)

                    # population sizes (affected 112)
                    setattr(self, f"inh_inter_pop_size_{side}",    300)
                    setattr(self, f"v1_pct_surviving_{side}",      1.0)
                    setattr(self, f"v2a_tonic_pop_size_{side}",    67)
                    setattr(self, f"v2a_bursting_pop_size_{side}", 0)
                    setattr(self, f"v1a_pop_size_{side}",          60)
                    setattr(self, f"v0c_pop_size_{side}",          15)
                    setattr(self, f"rc_pop_size_{side}",           30)
                    setattr(self, f"num_motor_neurons_{side}",    107)

                    if args['v0d_commissural_degeneration'] == 1: 

                        attr_name = f"w_custom_v0d_rg_inh_{side}_mean"
                        baseline_v0d_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v0d_strength * 0.95)  # 5% loss in synaptic strength from V0d at p112 
                        setattr(self, f"sparsity_custom_v0d_rg_inh_{side}", 0.08)  # V0d sparsity baseline 0.1275, reducing the sparsity 37.25% loss at p112 
                        setattr(self, f"v0d_pop_size_{side}", 48)                  # V0d pop size baseline 64, further reducing inhibitory pool

                    if args['v3_commissural_hyperexcitation'] == 1: 

                        attr_name = f"w_custom_v3_rg_{side}_mean"
                        baseline_v3_strength = getattr(self, attr_name)
                        setattr(self, attr_name, baseline_v3_strength * 1.25)  # 25% increase in synaptic strength from V3

                    
                    




        #Shared synaptic strengths across timepoints and frequencies
        self.w_rgexc_multiplier = 1.2 #Use this to change the RG balance through excitatory weights
        self.ratio_exc_inh = 4
        self.w_rg = 0.6 #0.6
        self.w_exc_mean = self.w_rg/self.ratio_exc_inh+self.w_rgexc_multiplier*self.w_rg 
        self.w_exc_std = 0.12 
        self.w_inh_mean = -self.w_rg if args['remove_inhibition'] == 0 else self.w_rg
        self.w_inh_std = 0.12 
        
        #Params for RG Layer - BASE CONNECTIONS 
        self.w_custom_rg_rg_mean = 0.05        
        self.w_custom_rg_rg_std = 0.01
        self.w_custom_rg_v2b_mean= 0.3      
        self.w_custom_rg_v2b_std= 0.01
        self.w_custom_v2b_rg_mean= -0.2      
        self.w_custom_v2b_rg_std= .01
        self.w_custom_rg_v1_mean= 0.35     
        self.w_custom_rg_v1_std= 0.035 
        self.w_custom_v1_rg_mean= -0.1*self.w_v1_multiplier if compensation == 1 else -0.1
        self.w_custom_v1_rg_std= .01
        
        # Asymmetric connections since these multiplier get affected in degeneration. 
        self.w_custom_rc_v1a_L_mean = -2.0*self.w_rc_multiplier_L*self.inh_weight_multiplier if compensation==1 else -2.0*self.inh_weight_multiplier #TESTING -0.024
        self.w_custom_rc_v1a_L_std = 0.09*self.inh_weight_multiplier 
        self.w_custom_rc_v1a_R_mean = -2.0*self.w_rc_multiplier_R*self.inh_weight_multiplier if compensation==1 else -2.0*self.inh_weight_multiplier #TESTING -0.024
        self.w_custom_rc_v1a_R_std = 0.09*self.inh_weight_multiplier 
        self.w_custom_rc_rc_L_mean = -4.0*self.w_rc_multiplier_L*self.inh_weight_multiplier if compensation==1 else -4.0*self.inh_weight_multiplier #TESTING -1.
        self.w_custom_rc_rc_L_std = 0.25*self.inh_weight_multiplier 
        self.w_custom_rc_rc_R_mean = -4.0*self.w_rc_multiplier_R*self.inh_weight_multiplier if compensation==1 else -4.0*self.inh_weight_multiplier #TESTING -1.
        self.w_custom_rc_rc_R_std = 0.25*self.inh_weight_multiplier 
        self.w_custom_rc_mn_L_mean= -3.*self.w_rc_multiplier_L*self.inh_weight_multiplier if compensation==1 else -3.*self.inh_weight_multiplier  #Use this to shape output - smooth curve   
        self.w_custom_rc_mn_L_std = 0.12*self.inh_weight_multiplier   
        self.w_custom_rc_mn_R_mean= -3.*self.w_rc_multiplier_R*self.inh_weight_multiplier if compensation==1 else -3.*self.inh_weight_multiplier  #Use this to shape output - smooth curve   
        self.w_custom_rc_mn_R_std = 0.12*self.inh_weight_multiplier   
        self.w_custom_v1a_mn_L_mean= -8*self.w_1a_multiplier_L*self.inh_weight_multiplier if compensation==1 else -8*self.inh_weight_multiplier #Use this to reduce co-activation TESTING -12
        self.w_custom_v1a_mn_L_std = .12*self.inh_weight_multiplier
        self.w_custom_v1a_mn_R_mean= -8*self.w_1a_multiplier_R*self.inh_weight_multiplier if compensation==1 else -8*self.inh_weight_multiplier #Use this to reduce co-activation TESTING -12
        self.w_custom_v1a_mn_R_std = .12*self.inh_weight_multiplier
        self.w_custom_v1a_v1a_L_mean= -4.0*self.w_1a_multiplier_L*self.inh_weight_multiplier if compensation==1 else -4.0*self.inh_weight_multiplier #TESTING -0.8
        self.w_custom_v1a_v1a_L_std =  0.12*self.inh_weight_multiplier
        self.w_custom_v1a_v1a_R_mean= -4.0*self.w_1a_multiplier_R*self.inh_weight_multiplier if compensation==1 else -4.0*self.inh_weight_multiplier #TESTING -0.8
        self.w_custom_v1a_v1a_R_std =  0.12*self.inh_weight_multiplier
        self.w_custom_v1_v2a_L_mean= -10*self.w_v1_multiplier_L*self.inh_weight_multiplier if compensation==1 else -5*self.inh_weight_multiplier #TESTING -1.495
        self.w_custom_v1_v2a_L_std = .045*self.inh_weight_multiplier
        self.w_custom_v1_v2a_R_mean= -10*self.w_v1_multiplier_R*self.inh_weight_multiplier if compensation==1 else -5*self.inh_weight_multiplier #TESTING -1.495
        self.w_custom_v1_v2a_R_std = .045*self.inh_weight_multiplier
        self.w_custom_v1_mn_L_mean= -4*self.w_v1_multiplier_L*self.inh_weight_multiplier if compensation==1 else -4*self.inh_weight_multiplier #TESTING 0.1
        self.w_custom_v1_mn_L_std = .001*self.inh_weight_multiplier  
        self.w_custom_v1_mn_R_mean= -4*self.w_v1_multiplier_R*self.inh_weight_multiplier if compensation==1 else -4*self.inh_weight_multiplier #TESTING 0.1
        self.w_custom_v1_mn_R_std = .001*self.inh_weight_multiplier  
        self.w_custom_v1_rg_L_mean = -0.1*self.w_v1_multiplier if compensation == 1 else -0.1
        self.w_custom_v1_rg_L_std = .01
        self.w_custom_v1_rg_R_mean = -0.1*self.w_v1_multiplier if compensation == 1 else -0.1
        self.w_custom_v1_rg_R_std = .01

        #Mean and Std for interneurons downstream of RG layer symmetrical connections - BASE CONNECTIONS 
        self.w_custom_rc_v1a_mean = -2*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -2*self.inh_weight_multiplier
        self.w_custom_rc_v1a_std = 0.09*self.inh_weight_multiplier 
        self.w_custom_rg_v1a_mean = 1.05*self.exc_weight_multiplier    
        self.w_custom_rg_v1a_std = 0.12*self.exc_weight_multiplier 
        self.w_custom_rc_rc_std = 0.25*self.inh_weight_multiplier     
        self.w_custom_mn_rc_mean= 8*self.exc_weight_multiplier  
        self.w_custom_rc_mn_mean= -3.*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -3.*self.inh_weight_multiplier  #Use this to shape output - smooth curve   
        self.w_custom_rc_mn_mean= -3.*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -3.*self.inh_weight_multiplier  #Use this to shape output - smooth curve   
        self.w_custom_rc_mn_std= 0.12*self.inh_weight_multiplier   
        self.w_custom_mn_rc_std= 0.12*self.exc_weight_multiplier  

        #       # BASE STD AND MEAN FOR CONNECTIONS  - might need some pruning... 
        self.w_custom_rg_v1a_mean = 1.05*self.exc_weight_multiplier  
        self.w_custom_rg_v1a_std = 0.12*self.exc_weight_multiplier 
        self.w_custom_rc_v1a_mean = -2*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -2*self.inh_weight_multiplier
        self.w_custom_rc_v1a_std = 0.09*self.inh_weight_multiplier 
        self.w_custom_rc_rc_mean = -4.0*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -4.0*self.inh_weight_multiplier
        self.w_custom_rc_rc_std = 0.25*self.inh_weight_multiplier  
        self.w_custom_rc_mn_mean= -3.*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -3.*self.inh_weight_multiplier  #Use this to shape output - smooth curve   
        self.w_custom_rc_mn_std= 0.12*self.inh_weight_multiplier   
        self.w_custom_mn_rc_mean= 8*self.exc_weight_multiplier  
        self.w_custom_mn_rc_std= 0.12*self.exc_weight_multiplier       
        self.w_custom_v1a_mn_mean= -8*self.w_1a_multiplier*self.inh_weight_multiplier if compensation==1 else -8*self.inh_weight_multiplier #Use this to reduce co-activation
        self.w_custom_v1a_mn_std= .12*self.inh_weight_multiplier
        self.w_custom_v1a_v1a_mean= -4.0*self.w_1a_multiplier*self.inh_weight_multiplier if compensation==1 else -4.*self.inh_weight_multiplier
        self.w_custom_v1a_v1a_std= 0.12*self.inh_weight_multiplier
        self.w_custom_v2b_v2a_mean= -10*self.inh_weight_multiplier
        self.w_custom_v2b_v2a_std= .045*self.inh_weight_multiplier     
        self.w_custom_v1_v2a_mean= -10*self.w_v1_multiplier*self.inh_weight_multiplier if compensation==1 else -5*self.inh_weight_multiplier
        self.w_custom_v1_v2a_std= .045*self.inh_weight_multiplier
        self.w_custom_v2b_mn_mean= -4*self.inh_weight_multiplier
        self.w_custom_v2b_mn_std= .001*self.inh_weight_multiplier     
        self.w_custom_v1_mn_mean= -4*self.w_v1_multiplier*self.inh_weight_multiplier if compensation==1 else -4*self.inh_weight_multiplier
        self.w_custom_v1_mn_std= .001*self.inh_weight_multiplier      
        self.w_custom_rg_v2a_mean = 1.*self.exc_weight_multiplier     
        self.w_custom_rg_v2a_std = 0.01*self.exc_weight_multiplier         
        self.w_custom_rg_v0c_mean = 1.*self.exc_weight_multiplier       
        self.w_custom_rg_v0c_std = 0.01*self.exc_weight_multiplier
        self.w_custom_v1a_mn_std= .12*self.inh_weight_multiplier
        self.w_custom_v1a_v1a_std= 0.12*self.inh_weight_multiplier
        self.w_custom_v2b_v2a_mean= -10*self.inh_weight_multiplier #TESTING -1.495
        self.w_custom_v2b_v2a_std= .045*self.inh_weight_multiplier     
        self.w_custom_v1_v2a_std= .045*self.inh_weight_multiplier
        self.w_custom_v2b_mn_mean= -4*self.inh_weight_multiplier     #TESTING 0.1
        self.w_custom_v2b_mn_std= .001*self.inh_weight_multiplier     
        self.w_custom_v1_mn_std= .001*self.inh_weight_multiplier      
        self.w_custom_rg_v2a_mean = 1.*self.exc_weight_multiplier   #TESTING 1. seeing if this weakens the rg layer
        self.w_custom_rg_v2a_std = 0.01*self.exc_weight_multiplier         
        self.w_custom_rg_v0c_mean = 1.*self.exc_weight_multiplier #TESTING 3.6        
        self.w_custom_rg_v0c_std = 0.01*self.exc_weight_multiplier
        self.w_custom_v0c_mn_mean= 10.56*self.exc_weight_multiplier if compensation==1 and self.days==112 else 5.28*self.exc_weight_multiplier
        self.w_custom_v0c_mn_std= 0.24*self.exc_weight_multiplier if compensation==1 and self.days==112 else 0.12*self.exc_weight_multiplier
        self.w_custom_v2a_mn_mean = 3.6*self.w_v2a_multiplier*self.exc_weight_multiplier if stabilized==1 else 3.6*self.exc_weight_multiplier          
        self.w_custom_v2a_mn_std = 0.12*self.exc_weight_multiplier         
        self.w_custom_v2a_selfexc_mean = 0.5*self.exc_weight_multiplier    
        self.w_custom_v2a_selfexc_std = 0.12*self.exc_weight_multiplier
        self.w_custom_v0v_v1_mean = 0.05*self.exc_weight_multiplier # V0v commissural connection v1 projecting over to excite v1 contralateral 
        self.w_custom_v0v_v1_std = 0.05*self.exc_weight_multiplier
        self.w_custom_v2a_v0v_mean =  0.05*self.exc_weight_multiplier # V0v ipsilateral connection v2a exciting v0v. 
        self.w_custom_v2a_v0v_std =  0.05*self.exc_weight_multiplier
        self.w_custom_rg_v2a_v0v_mean = 0.05 * self.exc_weight_multiplier  # V0v ipsilateral SEPERATE CONNECTION 
        self.w_custom_rg_v2a_v0v_std = 0.05 * self.exc_weight_multiplier
        self.w_custom_v1_rg_v0v_mean = 0.05  * self.inh_weight_multiplier # contralateral v1 inhibiting rg SEPERATE CONNECTION for V0v
        self.w_custom_v1_rg_v0v_std = 0.05 * self.inh_weight_multiplier
        self.w_custom_v0v_rg_inh_mean = 1.5*self.inh_weight_multiplier  # v0v inhibiting rg inhib flex. 
        self.w_custom_v0v_rg_inh_std = 1.5*self.inh_weight_multiplier 
        self.w_custom_v0c_mnp_flx_mean = 1.25*self.exc_weight_multiplier 
        self.w_custom_v0c_mnp_flx_std = 1.25*self.exc_weight_multiplier 
        self.w_custom_v0c_mnp_ext_mean = 1.25*self.exc_weight_multiplier 
        self.w_custom_v0c_mnp_ext_std = 1.25*self.exc_weight_multiplier 
        self.w_custom_rc_rc_mean = -4.0*self.w_rc_multiplier*self.inh_weight_multiplier if compensation==1 else -4.0*self.inh_weight_multiplier 

        #  rg v0d ipsilateral connection mean and std 
        self.w_custom_rg_v0d_R_mean = 0.4*self.exc_weight_multiplier   # rg exciting the v0d. Defined and Validated.
        self.w_custom_rg_v0d_R_std = 0.04*self.exc_weight_multiplier
        self.w_custom_rg_v0d_L_mean = 0.4*self.exc_weight_multiplier
        self.w_custom_rg_v0d_L_std = 0.04*self.exc_weight_multiplier
        self.w_custom_rg_v0d_mean = 0.4*self.exc_weight_multiplier
        self.w_custom_rg_v0d_std = 0.04*self.exc_weight_multiplier

        self.sparsity_custom_rg_v0d = 0.3 # rg exciting the v0d. Defined and Validated.
        self.sparsity_custom_rg_v0d_R = 0.3   
        self.sparsity_custom_rg_v0d_L = 0.3

        # sparsity params v3 
        self.sparsity_custom_v3_rg_L = 0.25 # v3 flx - flx sparsity params 
        self.sparsity_custom_v3_rg_R = 0.25
        self.sparsity_custom_v3_rg = 0.25        
        self.sparsity_custom_rg_v3_R = 0.15
        self.sparsity_custom_rg_v3_L = 0.15


       
        #Shared synaptic connectivity % across timepoints and frequencies symmetric connections 
        self.sparsity_rg = 0.03 #0.03 connectivity within RGs                                           
        self.sparsity_rg1_rg2 = 0.01 #0.01 connectivity directly between RGS          
        self.sparsity_custom_rg_rg = 0.01                                    
        self.sparsity_custom_rg_v1 = self.sparsity_rg_v1 #0.01                               
        self.sparsity_custom_rg_v2b = self.sparsity_rg_v2b #0.01      
        self.sparsity_custom_rg_v2a = 0.5 #0.5 
        self.sparsity_custom_rg_v1a = 0.7 #0.7    
        self.sparsity_custom_v2b_rg = self.sparsity_v2b_rg #0.01                           
        self.sparsity_custom_v2b_v2a = self.sparsity_v1s_outside_rg_layer 
        self.sparsity_custom_v2b_mn = self.sparsity_v1s_outside_rg_layer 
        self.sparsity_custom_rg_v0c = 0.5 #0.5         
        self.sparsity_custom_v0c_mn = 0.5 #0.5         
        self.sparsity_custom_v2a_mn = 0.5 #0.5       
        self.sparsity_custom_mn_rc = 0.5 #0.5
        self.sparsity_custom_v0c_mnp_flx = 0.5
        self.sparsity_custom_v0c_mnp_ext = 0.5 
        self.sparsity_custom_v0v_v1 = 0.5 
        self.sparsity_custom_rg_v2a_v0v = 0.5 
        self.sparsity_custom_v1_rg_v0v = 0.5    
        self.sparsity_custom_v0v_rg_inh = 0.25 # V0v sparsity connections 
        self.sparsity_custom_v2a_v0v = 0.25 
        
        # Map base synapses to L/R suffixes, this ensures that weights are kept from both left and right. ensuring healthy functioning. 
        base_to_lr = [

            "custom_rg_v2a_v0v",
            "custom_v2a_v0v",
            "custom_v0v_v1",
            "custom_v1_rg_v0v",
            # ----------------------------
            # RG → interneurons / MN
            # ----------------------------
            "custom_rg_v2a",
            "custom_rg_v1a",
            "custom_rg_v0c",
            "custom_rg_v1",
            "custom_rg_v2b",
            "custom_rg_rg",

            # ----------------------------
            # Interneurons → RG
            # ----------------------------
            "custom_v2b_rg",

            # ----------------------------
            # Interneurons → MN
            # ----------------------------
            "custom_v2a_mn",
            "custom_v2b_mn",
            "custom_v0c_mn",

            
            "custom_v2b_v2a",

        ]
        for base in base_to_lr:
            
            # weights
            setattr(self, f"w_{base}_L_mean", getattr(self, f"w_{base}_mean"))
            setattr(self, f"w_{base}_L_std",  getattr(self, f"w_{base}_std"))
            setattr(self, f"w_{base}_R_mean", getattr(self, f"w_{base}_mean"))
            setattr(self, f"w_{base}_R_std",  getattr(self, f"w_{base}_std"))

                # sparsity
            setattr(self, f"sparsity_{base}_L", getattr(self, f"sparsity_{base}"))
            setattr(self, f"sparsity_{base}_R", getattr(self, f"sparsity_{base}"))
         # End of mapping code. 


      
        #Rise and decay times for synapses
        inh_syn_multiplier = 1 if args['slow_syn_dyn']==0 else 1.6 
        exc_syn_multiplier = 1 if args['slow_syn_dyn']==0 else 1.6 
        self.tau_syn_e_rise = 0.2              
        self.tau_syn_e_decay = 1.0 
        self.tau_syn_e_rise_rc = 0.2 * exc_syn_multiplier #set Exc tau for effect from MNs to RCs           
        self.tau_syn_e_decay_rc = 1.0 * exc_syn_multiplier #set Exc tau for effect from MNs to RCs
        self.tau_syn_i_rise = 0.5 
        self.tau_syn_i_decay = 20.0
        self.tau_syn_i_rise_mn = 0.5 * inh_syn_multiplier #set Inh tau for effect from RCs, 1as to MNs 
        self.tau_syn_i_decay_mn = 20.0 * inh_syn_multiplier #set Inh tau for effect from RCs, 1as to MNs
        print('Excitatory synaptic rise/decay (RC)',self.tau_syn_e_rise_rc,self.tau_syn_e_decay_rc)
        print('Inhibitory synaptic rise/decay (MN)',self.tau_syn_i_rise_mn,self.tau_syn_i_decay_mn)
        
        #Shared population characteristics across timepoints and frequencies
        self.rg_pop_neurons= 1000   #1000
        self.rg_ext_exc_pct_tonic= 0.9 #0.9     
        self.rg_ext_inh_pct_tonic= 0.9 #0.9
        self.inh_inter_pct_tonic= 1. 
        
        # Frequency test (baseline=1)
        if self.freq_test==1:            
            self.I_e_bursting_mean = 300 #pA 300 normal 
            self.I_e_tonic_mean = 480 #pA 480 normal    
            self.rg_flx_exc_pct_tonic= 0.7   #0.7           
            self.rg_flx_inh_pct_tonic= 0.7   #0.7 
        elif self.freq_test==2:
            self.I_e_bursting_mean = 450 #pA    1pct 400  
            self.I_e_tonic_mean = 720 #pA       1pct 640    
            self.I_e_tonic_inh_mean = -450       # for the descending V0d inh drive    
            self.rg_flx_exc_pct_tonic= 0.7     #0.7         
            self.rg_flx_inh_pct_tonic= 0.7     #0.7                   
        
        # fb is not changed in degeneration no need for _L and _R here. 
        self.I_fb_bursting_mean = self.I_e_bursting_mean* self.fb_multiplier      
        self.I_fb_tonic_mean = self.I_e_tonic_mean * self.fb_multiplier 
        
        #Set parameters for network
        if len(self.argv) > 1:
            self.rng_seed = int(self.argv[1])
        else:
            self.rng_seed = self.args['seed']

        #self.rng_seed = np.random.randint(10**7) if args['seed'] == 0 else args['seed'] #set seed for NEST 	
        self.time_resolution = args['delta_clock'] 		#equivalent to "delta_clock"
        
        #self.rng_seed = np.random.randint(10**7) if args['seed'] == 0 else args['seed'] #set seed for NEST 	
        self.time_resolution = args['delta_clock'] 		#equivalent to "delta_clock"
        self.num_inh_inter_tonic_v2b = round(self.inh_inter_pop_size*self.inh_inter_pct_tonic)
        self.num_inh_inter_bursting_v2b = self.inh_inter_pop_size-self.num_inh_inter_tonic_v2b
        self.num_inh_inter_tonic_v1 = int(self.num_inh_inter_tonic_v2b*self.v1_pct_surviving)
        self.num_inh_inter_bursting_v1 = int(self.num_inh_inter_bursting_v2b*self.v1_pct_surviving)
        self.exc_neurons_count = int(np.round(self.rg_pop_neurons * (self.ratio_exc_inh / (self.ratio_exc_inh + 1)))) # N_E = N*(r / (r+1))
        self.inh_neurons_count = int(np.round(self.rg_pop_neurons * ( 1 / (self.ratio_exc_inh + 1)))) # N_I = N*(1 / (r+1))
        self.flx_exc_tonic_count = round(self.exc_neurons_count*self.rg_flx_exc_pct_tonic)
        self.flx_exc_bursting_count = self.exc_neurons_count-self.flx_exc_tonic_count
        self.flx_inh_tonic_count = round(self.inh_neurons_count*self.rg_flx_inh_pct_tonic)
        self.flx_inh_bursting_count = self.inh_neurons_count-self.flx_inh_tonic_count
        self.ext_exc_tonic_count = round(self.exc_neurons_count*self.rg_ext_exc_pct_tonic)
        self.ext_exc_bursting_count = self.exc_neurons_count-self.ext_exc_tonic_count
        self.ext_inh_tonic_count = round(self.inh_neurons_count*self.rg_ext_inh_pct_tonic)
        self.ext_inh_bursting_count = self.inh_neurons_count-self.ext_inh_tonic_count

        self.num_inh_inter_tonic_v2b_L = round(self.inh_inter_pop_size_L*self.inh_inter_pct_tonic)
        self.num_inh_inter_bursting_v2b_L = self.inh_inter_pop_size_L-self.num_inh_inter_tonic_v2b_L
        self.num_inh_inter_tonic_v2b_R = round(self.inh_inter_pop_size_R*self.inh_inter_pct_tonic)
        self.num_inh_inter_bursting_v2b_R = self.inh_inter_pop_size_R-self.num_inh_inter_tonic_v2b_R
        self.num_inh_inter_tonic_v1_L = int(self.num_inh_inter_tonic_v2b_L*self.v1_pct_surviving_L)
        self.num_inh_inter_tonic_v1_R = int(self.num_inh_inter_tonic_v2b_R*self.v1_pct_surviving_R)
        self.num_inh_inter_bursting_v1_L = int(self.num_inh_inter_bursting_v2b_L*self.v1_pct_surviving_L)
        self.num_inh_inter_bursting_v1_R = int(self.num_inh_inter_bursting_v2b_R*self.v1_pct_surviving_R)
        self.exc_neurons_count = int(np.round(self.rg_pop_neurons * (self.ratio_exc_inh / (self.ratio_exc_inh + 1)))) # N_E = N*(r / (r+1))
        self.inh_neurons_count = int(np.round(self.rg_pop_neurons * ( 1 / (self.ratio_exc_inh + 1)))) # N_I = N*(1 / (r+1))
        self.flx_exc_tonic_count = round(self.exc_neurons_count*self.rg_flx_exc_pct_tonic)
        self.flx_exc_bursting_count = self.exc_neurons_count-self.flx_exc_tonic_count
        self.flx_inh_tonic_count = round(self.inh_neurons_count*self.rg_flx_inh_pct_tonic)
        self.flx_inh_bursting_count = self.inh_neurons_count-self.flx_inh_tonic_count
        self.ext_exc_tonic_count = round(self.exc_neurons_count*self.rg_ext_exc_pct_tonic)
        self.ext_exc_bursting_count = self.exc_neurons_count-self.ext_exc_tonic_count
        self.ext_inh_tonic_count = round(self.inh_neurons_count*self.rg_ext_inh_pct_tonic)
        self.ext_inh_bursting_count = self.inh_neurons_count-self.ext_inh_tonic_count
        self.sim_time = args['t_steps']         #time in ms
        
        #Initialize neuronal parameters
        self.V_th_mean_tonic = -50.0 #mV  
        self.V_th_std_tonic = 1.0 #mV
        self.V_th_v1v2b_mean_tonic = -50.0 #mV  
        self.V_th_v1v2b_std_tonic = 1.0 #mV
        self.V_th_mean_bursting = -51.0 #mV         
        self.V_th_std_bursting = 1.0 #mV
        self.V_m_mean = -60.0 #mV 
        self.V_m_std = 10.0 #mV
        self.C_m_bursting_mean = 600.0 #pF         
        self.C_m_bursting_std = 80.0 #pF     
        self.C_m_tonic_mean = 200.0 #pF            
        self.C_m_tonic_std = 40.0 #pF
        self.C_m_mn_tonic_mean = 200.0 #pF         
        self.C_m_mn_tonic_std = 40.0 #pF 
        self.C_m_v1v2b_tonic_mean = 200.0 #pF 
        self.C_m_v1v2b_tonic_std = 40.0 #pF        
        self.t_ref_mean = 9.0 #ms               
        self.t_ref_std = 0.2 #ms
        self.t_ref_bursting_mean = 3.0 #ms 
        self.t_ref_bursting_std = 0.2 #ms
        
        # Importing Args from configuration_run_nest.yaml 
        self.synaptic_delay = 2. #args['synaptic_delay']
        self.rgs_connected = args['rgs_connected']
        self.v1v2b_mn_connected = args['v1v2b_mn_connected']
        self.remove_descending_drive = args['remove_descending_drive']
        self.slow_syn_bias = args['slow_syn_bias']

        # Commissural Interneurons 
        self.low_locomotion_v0d_left = args['low_locomotion_v0d_left']  # v0d  
        self.low_locomotion_v0d_right = args['low_locomotion_v0d_right']
        self.low_locomotion_v3_right = args['low_locomotion_v3_right'] # v3 
        self.low_locomotion_v3_left = args['low_locomotion_v3_left'] 
        
        # asymmetiric degeneration params 
        self.v0d_degeneration_plot = args['v0d_degeneration_plot']
        self.asymmetric_onset = args['asymmetric_onset']
        self.asym_side = args['asym_side']
        self.v0d_commissural_degeneration = args['v0d_commissural_degeneration']
        self.v3_commissural_hyperexcitation = args['v3_commissural_hyperexcitation']

        # online ramp params 
        self.w_start = args['w_start']
        self.w_end = args['w_end']
        self.online_ramp_experiment = args['online_ramp_experiment']
        self.online_ramp_weight = args['online_ramp_weight']
        self.online_ramp_weight_experiment = args['online_ramp_weight_experiment']
        self.online_ramp_drive_experiment = args['online_ramp_drive_experiment']
        self.online_stepwise_drive_targets = args['online_stepwise_drive_targets']
        self.d_start = args['d_start']
        self.d_end = args['d_end']

        # metric summary table 
        self.save_metric_summary_table = args['save_metric_summary_table']
     
        # Arguments for scaling experiments - online ramp
        self.ramp_duration = args['ramp_duration']
        self.online_ramp_experiment = args['online_ramp_experiment']
        self.online_ramp_weight = args['online_ramp_weight']

        # Arguments for scaling experiments - online stepwise 
        self.online_stepwise_experiment = args['online_stepwise_experiment']
        self.online_stepwise_weight = args['online_stepwise_weight']
        self.stepwise_weight_experiment = args['stepwise_weight_experiment']
        self.stepwise_drive_experiment = args['stepwise_drive_experiment']
        self.online_stepwise_drive_targets = args['online_stepwise_drive_targets']
        self.step_scales = args['step_scales']
        self.step_hold_ms = args['step_hold_ms']

        # arguments for saving heatlhy regime metrics 
        self.healthy_regime_metrics = args['healthy_regime_metrics']

        #Feedback
        self.num_pgs = 100
        self.fb_rg_flx = args['fb_rg_flx']
        self.fb_rg_ext = args['fb_rg_ext']
        self.fb_v2b = args['fb_v2b']
        self.fb_v1 = args['fb_v1']
        self.fb_1a_flx = args['fb_1a_flx']
        self.fb_1a_ext = args['fb_1a_ext']
        self.sim_fb_freq = args['sim_fb_freq']

        print('Running freq test ',self.freq_test,', Mean desc current (T,B), Mean fb current (T,B): ',self.I_e_tonic_mean,self.I_e_bursting_mean,self.I_fb_tonic_mean,self.I_fb_bursting_mean)
        self.I_e_bursting_std = 0.25*self.I_e_bursting_mean #pA 
        self.I_e_tonic_std = 0.25*self.I_e_tonic_mean #pA
        #std for the descending inh input for V0d. 
        self.I_e_tonic_inh_L_std = 0.01 * abs(self.I_e_tonic_inh_L_mean) # both pA.
        self.I_e_tonic_inh_R_std = 0.01 * abs(self.I_e_tonic_inh_R_mean) # since both are negative ints 
        self.I_e_tonic_v3_L_std = 0.01 * self.I_e_tonic_v3_L_mean # positive ints 
        self.I_e_tonic_v3_R_std = 0.01 * self.I_e_tonic_v3_R_mean 
        self.I_fb_bursting_std = 0.25*self.I_fb_bursting_mean #pA 
        self.I_fb_tonic_std = 0.25*self.I_fb_tonic_mean #pA
        self.noise_std_dev_tonic = self.I_e_tonic_mean #pA 
        self.noise_std_dev_bursting = self.I_e_bursting_mean #pA
        print('Noise standard deviation (T,B) ',self.noise_std_dev_tonic,self.noise_std_dev_bursting)

        #Set data evaluation parameters
        self.convstd_rate = args['convstd_rate']
        self.convstd_pca = args['convstd_pca']
        self.chop_edges_amount = args['chop_edges_amount']
        self.remove_mean = args['remove_mean']
        self.high_pass_filtered = args['high_pass_filtered']
        self.downsampling_convolved = args['downsampling_convolved']
        self.remove_silent = args['remove_silent']
        self.PCA_components = args['PCA_components']
        self.calculate_balance = args['calculate_balance']               
        self.raster_plot = args['raster_plot']
        self.rate_coded_plot = args['rate_coded_plot']
        self.spike_distribution_plot = args['spike_distribution_plot']
        self.membrane_potential_plot = args['membrane_potential_plot']
        self.isf_output = args['isf_output']
        self.time_window = args['smoothing_window']
        self.phase_ordered_plot = args['phase_ordered_plot']
        self.heatmap_recruitment_plot = args['heatmap_recruitment_plot']
        self.save_v0d_pops = args['save_v0d_pops']
        self.save_rg_v1_pops = args['save_rg_v1_pops']
        self.save_v3_pops = args['save_v3_pops']
        self.save_as_svg = args['save_as_svg']
        self.overlap_plot = args['overlap_plot']

        #Set spike detector parameters 
        self.sd_params = {"withtime" : True, "withgid" : True, 'to_file' : False, 'flush_after_simulate' : False, 'flush_records' : True}
        
        self.conn_dict_custom_rg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_rg}		
        self.conn_dict_custom_rg1_rg2 = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_rg1_rg2}
        self.conn_dict_custom_v1a_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_mn}
        self.conn_dict_custom_v1a_v1a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_v1a}
        self.conn_dict_custom_rg_v1 = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1}
        self.conn_dict_custom_rg_v2b = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2b}
        self.conn_dict_custom_rg_v2a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2a}
        self.conn_dict_custom_rg_v1a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1a}
        self.conn_dict_custom_v1_rg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_rg}
        self.conn_dict_custom_v2b_rg = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_rg}
        self.conn_dict_custom_v1_v2a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_v2a}
        self.conn_dict_custom_v2b_v2a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_v2a}
        self.conn_dict_custom_v1_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_mn}
        self.conn_dict_custom_v2b_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_mn}
        self.conn_dict_custom_rg_v0c = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0c}
        self.conn_dict_custom_v0c_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0c_mn}
        self.conn_dict_custom_v2a_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2a_mn}
        self.conn_dict_custom_rc_v1a = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_v1a}
        self.conn_dict_custom_rc_rc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_rc}
        self.conn_dict_custom_rc_mn = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_mn}
        self.conn_dict_custom_mn_rc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_mn_rc}
        self.conn_dict_custom_v0v_v1 = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v0v_v1}
        self.conn_dict_custom_rg_v2a_v0v = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_rg_v2a_v0v}
        self.conn_dict_custom_v1_rg_v0v = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v1_rg_v0v}
        self.conn_dict_custom_rg_v0d = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0d}
        self.conn_dict_custom_v2a_v0v = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2a_v0v}       
        self.conn_dict_custom_v0v_rg_inh = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0v_rg_inh}  
        self.conn_dict_custom_v0c_mnp_flx = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0c_mnp_flx}       
        self.conn_dict_custom_v0c_mnp_ext = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0c_mnp_ext} 
        self.conn_dict_custom_v0d_rg_inh = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0d_rg_inh}          
     
        # Conn dict asymmetric connection params left. 
        self.conn_dict_custom_v0d_rg_inh_L = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0d_rg_inh_L}
        self.conn_dict_custom_rg_v0d_L = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0d_L}
        self.conn_dict_custom_rg_v2a_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2a_L}
        self.conn_dict_custom_v2a_mn_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2a_mn_L}
        self.conn_dict_custom_rg_v1a_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1a_L}
        self.conn_dict_custom_rg_v0c_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0c_L}
        self.conn_dict_custom_v0c_mn_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0c_mn_L}
        self.conn_dict_custom_rg_v2b_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2b_L}
        self.conn_dict_custom_rg_v1_L    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1_L}
        self.conn_dict_custom_v2b_rg_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_rg_L}
        self.conn_dict_custom_v2b_v2a_L  = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_v2a_L}
        self.conn_dict_custom_v2b_mn_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_mn_L}
        self.conn_dict_custom_v1_rg_L    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_rg_L}
        self.conn_dict_custom_v1_v2a_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_v2a_L}
        self.conn_dict_custom_v1_mn_L    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_mn_L}
        self.conn_dict_custom_rc_mn_L    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_mn_L}
        self.conn_dict_custom_rc_v1a_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_v1a_L}
        self.conn_dict_custom_rc_rc_L    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_rc_L}
        self.conn_dict_custom_v1a_mn_L   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_mn_L}
        self.conn_dict_custom_v1a_v1a_L = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_v1a_L}
        self.conn_dict_custom_rg_rg_L    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_rg_L}
        self.conn_dict_custom_rg_v2a_v0v_L = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_rg_v2a_v0v_L}
        self.conn_dict_custom_v2a_v0v_L = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v2a_v0v_L}
        self.conn_dict_custom_v0v_v1_L = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v0v_v1_L}
        self.conn_dict_custom_v1_rg_v0v_L = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v1_rg_v0v_L}

        # conn dict asymmetric connections params right. 
        self.conn_dict_custom_v0d_rg_inh_R = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0d_rg_inh_R}
        self.conn_dict_custom_rg_v0d_R = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0d_R}
        self.conn_dict_custom_rg_v2a_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2a_R}
        self.conn_dict_custom_v2a_mn_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2a_mn_R}
        self.conn_dict_custom_rg_v1a_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1a_R}
        self.conn_dict_custom_rg_v0c_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v0c_R}
        self.conn_dict_custom_v0c_mn_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v0c_mn_R}
        self.conn_dict_custom_rg_v2b_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v2b_R}
        self.conn_dict_custom_rg_v1_R    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_v1_R}
        self.conn_dict_custom_v2b_rg_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_rg_R}
        self.conn_dict_custom_v2b_v2a_R  = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_v2a_R}
        self.conn_dict_custom_v2b_mn_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v2b_mn_R}
        self.conn_dict_custom_v1_rg_R    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_rg_R}
        self.conn_dict_custom_v1_v2a_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_v2a_R}
        self.conn_dict_custom_v1_mn_R    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1_mn_R}
        self.conn_dict_custom_rc_mn_R    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_mn_R}
        self.conn_dict_custom_rc_v1a_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_v1a_R}
        self.conn_dict_custom_rc_rc_R    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rc_rc_R}
        self.conn_dict_custom_v1a_mn_R   = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_mn_R}
        self.conn_dict_custom_v1a_v1a_R = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_v1a_v1a_R}
        self.conn_dict_custom_rg_rg_R    = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_custom_rg_rg_R}
        self.conn_dict_custom_rg_v2a_v0v_R = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_rg_v2a_v0v_R}
        self.conn_dict_custom_v2a_v0v_R = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v2a_v0v_R}
        self.conn_dict_custom_v0v_v1_R = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v0v_v1_R}
        self.conn_dict_custom_v1_rg_v0v_R = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v1_rg_v0v_R}
        self.conn_dict_custom_rg_v3_L = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_rg_v3_L}
        self.conn_dict_custom_rg_v3_R = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_rg_v3_R}
        self.conn_dict_custom_v3_rg_L = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v3_rg_L}
        self.conn_dict_custom_v3_rg_R = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v3_rg_R}
        self.conn_dict_custom_v3_rg = {'rule': 'pairwise_bernoulli', 'p':self.sparsity_custom_v3_rg}

        #Set multimeter parameters
        self.mm_params = {'interval': 1., 'record_from': ['V_m']}

        #Set noise parameters
        self.noise_params_tonic = {"dt": self.time_resolution, "std":self.noise_std_dev_tonic}
        self.noise_params_bursting = {"dt": self.time_resolution, "std":self.noise_std_dev_bursting}
   
        
        ################
        # Save results #
        ################
        if self.args.get("save_results", 0) and not self.args.get("optimizing", 0):

            if hasattr(neural_network, "_save_initialized"):
                del neural_network._save_initialized

            if len(self.argv) > 1:
                id_ = f"{int(self.argv[1])}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            else:
                id_ = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            base = pathlib.Path("saved_simulations").resolve()
            path = base / self.run_tag / id_
            pathFigures = path / "Figures"

            # Don't create folders yet — defer until first actual save
            neural_network._id_ = id_
            neural_network._path = str(path)
            neural_network._pathFigures = str(pathFigures)
            neural_network._args_to_save = self.args  # save for later


            self.id_ = neural_network._id_
            print("ID:", self.id_)
            self.path = neural_network._path
            print("SAVE PATH:", self.path)
            self.pathFigures = neural_network._pathFigures

        else:
            self.id_ = None
            self.path = None
            self.pathFigures = None
