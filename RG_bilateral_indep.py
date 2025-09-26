import nest
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os 

# ---------------- LOAD CONFIG ----------------
save_dir = "/Users/angusgray/Desktop/Dissertation/visualisation/figure_output/RG_bilateral"
os.makedirs(save_dir, exist_ok=True)


with open("/Users/angusgray/Desktop/Dissertation/nest_models/NEST_inbuild/config/config.yaml") as f:
    config = yaml.safe_load(f)

# ---------------- NEST SETUP ----------------
nest.ResetKernel()
nest.SetKernelStatus({
    "resolution": config["dt"],
    "rng_seed": int(config["seed"])
})

# ---------------- POPULATIONS ----------------
pops = {}
recorders = {}

def make_pop(name):
    """Create population and spike recorder"""
    pop_cfg = config["populations"][name]
    N = pop_cfg["size"]
    params = dict(config["params"][pop_cfg["type"]])  # ensure plain dict
    neurons = nest.Create("aeif_cond_alpha", N, params=params)
    rec = nest.Create("spike_recorder")
    nest.Connect(neurons, rec)
    pops[name] = neurons
    recorders[name] = rec


# Create only RG groups (left + right, flexor + extensor)
for name in config["populations"]:
    if name.startswith("RG_Flx") or name.startswith("RG_Ext"):
        make_pop(name)

# ---------------- INPUTS ----------------
pg_L = nest.Create("poisson_generator", params={"rate": config["poisson_rate"]})
pg_R = nest.Create("poisson_generator", params={"rate": config["poisson_rate"]})

for name in pops:
    pop_cfg = config["populations"][name]

    # Poisson input (optional, if you still want it)
    if "L_" in name:
        nest.Connect(pg_L, pops[name],
                     syn_spec={"weight": config["pg_weight"],
                               "delay": config["pg_delay"]})
    elif "R_" in name:
        nest.Connect(pg_R, pops[name],
                     syn_spec={"weight": config["pg_weight"],
                               "delay": config["pg_delay"]})

    # DC drive from Table A1
    if "drive" in pop_cfg:
        dc = nest.Create("dc_generator", params={"amplitude": pop_cfg["drive"]})
        nest.Connect(dc, pops[name])

# ---------------- IPSILATERAL INHIBITION ----------------
# Flexor <-> Extensor inhibition on each side
def connect_ipsi(side):
    F_groups = [k for k in pops if k.startswith(f"{side}_RG_F")]
    E_groups = [k for k in pops if k.startswith(f"{side}_RG_E")]
    for F in F_groups:
        for E in E_groups:
            nest.Connect(pops[F], pops[E],
                         conn_spec={"rule": "pairwise_bernoulli", "p": config["p_conn"]},
                         syn_spec={"weight": config["weight_ipsi"], "delay": config["delay_ipsi"]})
            nest.Connect(pops[E], pops[F],
                         conn_spec={"rule": "pairwise_bernoulli", "p": config["p_conn"]},
                         syn_spec={"weight": config["weight_ipsi"], "delay": config["delay_ipsi"]})

connect_ipsi("L")
connect_ipsi("R")

# ---------------- SIMULATION ----------------
nest.Simulate(config["simtime"])


# --- quick diagnostics after simulation ---
def pop_spike_counts(rec):
    ev = nest.GetStatus(rec, "events")[0]
    times = np.array(ev.get("times", []))
    senders = np.array(ev.get("senders", []))
    return len(times)

print("\n=== POPULATION SPIKE SUMMARY ===")
for name in pops:
    cnt = pop_spike_counts(recorders[name])
    # mean rate (overall)
    N = config["populations"][name]["size"]
    mean_rate = 0.0
    if cnt > 0:
        # total simtime from config
        mean_rate = cnt / (config["simtime"]/1000.0) / N
    print(f"{name:20s} | spikes={cnt:6d} | mean_rate={mean_rate:6.2f} Hz per neuron | size={N}")

# print raw drive/inhibition params to cross-check
print("\nDRIVES and INHIBITORY SETTINGS:")
for name in pops:
    print(f"{name:20s} drive={config['populations'][name].get('drive',0)}")
print(f"weight_ipsi={config['weight_ipsi']}, p_conn={config['p_conn']}, delay_ipsi={config['delay_ipsi']}")



def compute_metrics_LR(baseF, baseE, recorders, populations, simtime, binsz=20.0, thresh=10.0):
    """
    Compute overlap, duty cycle, phase lag, and mean firing rate for
    left and right populations.

    Args:
        baseF, baseE: base names of flexor/extensor populations, e.g., "RG_Flx_exc"
        recorders: dict of NEST spike recorders
        populations: config["populations"]
        simtime: total simulation time (ms)
        binsz: histogram bin size (ms)
        thresh: threshold for defining active bins (Hz)

    Returns:
        dict with keys 'L' and 'R', each containing the metric dict
    """
    results = {}
    for side in ["L", "R"]:
        recF_name = f"{baseF}_{side}"
        recE_name = f"{baseE}_{side}"

        if recF_name not in recorders or recE_name not in recorders:
            print(f"Warning: Recorders for {recF_name} or {recE_name} not found.")
            results[side] = None
            continue

        recF = recorders[recF_name]
        recE = recorders[recE_name]

        Nf = populations[recF_name]["size"]
        Ne = populations[recE_name]["size"]

        evF = nest.GetStatus(recF, "events")[0]; timesF = np.array(evF["times"])
        evE = nest.GetStatus(recE, "events")[0]; timesE = np.array(evE["times"])

        bins = np.arange(0, simtime + binsz, binsz)
        cF, _ = np.histogram(timesF, bins=bins)
        cE, _ = np.histogram(timesE, bins=bins)

        rF = cF / (binsz/1000.0) / Nf
        rE = cE / (binsz/1000.0) / Ne

        both = (rF > thresh) & (rE > thresh)
        overlap_frac = both.sum() / len(both)
        dutyF = (rF > thresh).sum() / len(rF)
        dutyE = (rE > thresh).sum() / len(rE)

        # rough phase lag: cross-correlation peak lag in bins
        xcorr = np.correlate((rF>thresh).astype(int)-0.5, (rE>thresh).astype(int)-0.5, mode='full')
        lag_bins = np.argmax(xcorr) - (len(rF)-1)
        lag_ms = lag_bins * binsz

        results[side] = {
            "overlap": overlap_frac,
            "dutyF": dutyF,
            "dutyE": dutyE,
            "lag_ms": lag_ms,
            "meanF": rF.mean(),
            "meanE": rE.mean()
        }

    return results


# Call metric function 
metrics = compute_metrics_LR("RG_Flx_exc", "RG_Ext_exc", recorders, config["populations"], config["simtime"], binsz=20.0, thresh=8.0)

print("EXCITATORY LEFT metrics:", metrics["L"])
print("EXCITATORY RIGHT metrics:", metrics["R"])

metrics = compute_metrics_LR("RG_Flx_inh", "RG_Ext_inh", recorders, config["populations"], config["simtime"], binsz=20.0, thresh=8.0)
print("INHIB LEFT metrics:", metrics["L"])
print("INHIB RIGHT metrics:", metrics["R"])

# ---------------- PLOTTING ----------------
def get_times(rec):
    ev = nest.GetStatus(rec, "events")[0]
    return np.array(ev["times"]), np.array(ev["senders"])

# Raster (grouped)
plt.figure(figsize=(12,6))
colors = {"F":"red","E":"blue","inh":"orange","exc":"green"}
offset = 0
for name in pops:
    ts, s = get_times(recorders[name])
    if ts.size > 0:
        color = "black"
        if "F" in name: color = "red"
        if "E" in name: color = "blue"
        if "inh" in name: color = "orange"
        if "exc" in name: color = "green"
        plt.scatter(ts, s + offset, s=1, c=color, label=name if offset==0 else None)
    offset += config["populations"][name]["size"] + 5
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID (offset by group)")
plt.title("Raster of RG populations")
plt.tight_layout()
fn = os.path.join(save_dir, "Raster Grouped.png")
plt.savefig(fn)
plt.show()
plt.close()
print(f"saved: {fn}")

# ---------------- Population rate function ----------------
def pop_rate(rec, N, simtime, binsz=20.0):
    ev = nest.GetStatus(rec, "events")[0]
    times = np.array(ev["times"])
    if len(times) == 0:
        return np.array([]), np.array([])
    bins = np.arange(0, simtime + binsz, binsz)
    counts, _ = np.histogram(times, bins=bins)
    rate = counts / (binsz / 1000.0) / N
    tcent = bins[:-1] + binsz / 2.0
    return tcent, rate


# ---------------- Plotting function ----------------
def plot_population_rates(baseF, baseE, recorders, populations, simtime, side, save_dir):
    """
    Plot population rates for flexor/extensor populations for a given side (L or R)
    """
    recF_name = f"{baseF}_{side}"
    recE_name = f"{baseE}_{side}"

    if recF_name not in recorders or recE_name not in recorders:
        print(f"Warning: Recorders for {recF_name} or {recE_name} not found.")
        return

    Nf = populations[recF_name]["size"]
    Ne = populations[recE_name]["size"]

    tF, rF = pop_rate(recorders[recF_name], Nf, simtime)
    tE, rE = pop_rate(recorders[recE_name], Ne, simtime)

    plt.figure(figsize=(8, 3))
    plt.plot(tF, rF, label=f"Flexor ({side} burst)", color="red")
    plt.plot(tE, rE, label=f"Extensor ({side} burst)", color="blue")
    plt.title(f"Example population rates ({'Left' if side=='L' else 'Right'} Hemicord)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate (Hz/neuron)")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(save_dir, f"Population_Rates_{'Left' if side=='L' else 'Right'}.png")
    plt.savefig(fn)
    plt.show()
    plt.close()
    print(f"saved: {fn}")


# ---------------- Generate plots ----------------
plot_population_rates("RG_Flx_exc", "RG_Ext_exc", recorders, config["populations"], config["simtime"], side="L", save_dir=save_dir)
plot_population_rates("RG_Flx_exc", "RG_Ext_exc", recorders, config["populations"], config["simtime"], side="R", save_dir=save_dir)
