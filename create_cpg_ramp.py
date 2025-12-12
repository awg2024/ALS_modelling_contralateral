
#!/usr/bin/env python

import nest
import numpy as np
import matplotlib.pyplot as plt
import start_simulation as ss
import set_network_params as netparams
import population_functions as popfunc
from ramp_circuitry import build_full_cpg_network
from ramp_cpg_utils import cpg_utils
import os
import time


# --------------------------------------
# INITIALISE NETWORK
# --------------------------------------
nn = netparams.neural_network()
nest.ResetKernel()
ss.nest_start()

# call in full network model.
print("Calling Full Network Model. ")
pops = build_full_cpg_network()

L_mnp1, L_mnp2 = pops["L_mnp1"], pops["L_mnp2"]
R_mnp1, R_mnp2 = pops["R_mnp1"], pops["R_mnp2"]

L_V0D, R_V0D = pops["L_V0D"], pops["R_V0D"]
L_V0V, R_V0V = pops["L_V0V"], pops["R_V0V"]

L_rg1, R_rg1 = pops["L_rg1"], pops["R_rg1"]
L_rg2, R_rg2 = pops["L_rg2"], pops["R_rg2"]

L_V2a, R_V2a = pops["L_exc1"], pops["R_exc1"]

# pops["L_exc2"]
# pops["R_exc2"] = R_V2a_2? 


#Check integrity of V0v and V0d creation. 
print("integrity of v0v and v0d after build_full_cpg_network")
print("L_V0V.v0v_all type:", type(L_V0V.v0v_all), "len:", len(L_V0V.v0v_all))
print("R_V0V.v0v_all type:", type(R_V0V.v0v_all), "len:", len(R_V0V.v0v_all))
print("L_V0D.v0d_all type:", type(L_V0D.v0d_all), "len:", len(L_V0D.v0d_all))
print("R_V0D.v0d_all type:", type(R_V0D.v0d_all), "len:", len(R_V0D.v0d_all))
time.sleep(10)

# --------------------------------------
# Identify V0D→RG synapses (both directions)
# --------------------------------------

print("Identifying specified ramp conditions.")

if nn.ramp_experimental_v0d_weights:
    src_L = L_V0D.v0d_bursting
    tgt_R = R_rg1.rg_exc_bursting
else:
    src_L = L_V0D.v0d_tonic
    tgt_R = R_rg1.rg_exc_tonic

# R→L
if nn.ramp_experimental_v0d_weights:
    src_R = R_V0D.v0d_bursting
    tgt_L = L_rg1.rg_exc_bursting
else:
    src_R = R_V0D.v0d_tonic
    tgt_L = L_rg1.rg_exc_tonic

# ------------------------------------------
# V0V L → R
# ------------------------------------------
if nn.ramp_experimental_v0v_weights:
    # HIGH locomotion → V0v bursting inhibits RG_tonic
    src_L_v0v = L_V0V.v0v_bursting
    tgt_R_v0v = R_rg1.rg_exc_tonic
else:
    # LOW locomotion → V0v tonic inhibits RG_bursting
    src_L_v0v = L_V0V.v0v_tonic
    tgt_R_v0v = R_rg1.rg_exc_bursting

# ------------------------------------------
# V0V R → L
# ------------------------------------------
if nn.ramp_experimental_v0v_weights:
    # HIGH locomotion
    src_R_v0v = R_V0V.v0v_bursting
    tgt_L_v0v = L_rg1.rg_exc_tonic
else:
    # LOW locomotion
    src_R_v0v = R_V0V.v0v_tonic
    tgt_L_v0v = L_rg1.rg_exc_bursting



conns_LR = nest.GetConnections(source=src_L, target=tgt_R)
conns_RL = nest.GetConnections(source=src_R, target=tgt_L)

#print(f"[DEBUG] Conns_LR = {conns_LR}")
#print(f"[DEBUG] Conns_RL = {conns_RL}")

baseline_LR = np.mean(nest.GetStatus(conns_LR, "weight"))
baseline_RL = np.mean(nest.GetStatus(conns_RL, "weight"))

print(f"[DEBUG] L→R baseline weight = {baseline_LR:.4f}")
print(f"[DEBUG] R→L baseline weight = {baseline_RL:.4f}")

# --------------------------------------
# RAMP PARAMETERS
# --------------------------------------
scales = [0.0, 0.5, 1.0, 2.0, 8.0]
sim_time = 2000
global_time = 0


# --------------------------------------
# MAIN LOOP
# --------------------------------------
for k, scale in enumerate(scales):

    new_LR = baseline_LR * scale
    new_RL = baseline_RL * scale

    nest.SetStatus(conns_LR, {"weight": new_LR})
    nest.SetStatus(conns_RL, {"weight": new_RL})

    print(f"[BLOCK {k}] Running Ramp Experiment - Scale: {scale:.2f}")
    nest.Simulate(sim_time)

    label = f"scale_{scale:.2f}"

    # CALL the V0d/V0v plotting utility
    cpg_utils(
        nn, popfunc,
        L_rg1, L_rg2, R_rg1, R_rg2,
        L_V2a, R_V2a,
        L_V0V, R_V0V,
        L_V0D, R_V0D,
        L_mnp1, L_mnp2, R_mnp1, R_mnp2,
        label
    )
