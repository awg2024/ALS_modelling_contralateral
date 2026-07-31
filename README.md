
# Bilateral CPG Model Simulating the Asymmetric Spread of ALS 

This repository contains a <b> spiking neural network (SNN) </b> implementation of a <b> bilateral central pattern generator (CPG) </b> model designed to simulate ipsilateral and commissural interneuron degeneration in ALS, extending the work of Strohmer et al. (2025).

The network is built in a modular and configurable way, allowing users to define and run a range of experiments and simulations through <i> configuration_run_nest.yaml </i>. This includes options to specify:

<ul>
<li> simulation duration </li>
<li> stepwise weight/drive experiments </li>
<li> ramped weight/drive experiments </li>
<li> selective activation or deactivation of commissural populations </li>
</ul>


Model parameters are defined in <i> set_network_params.py </i>, where key properties such as connection strengths and connection sparsity are specified.

The repository also includes evaluation scripts for analysing neural activity using a variety of quantitative approaches.

Simulations and experiments can be initialised and executed using the bash scripts provided in the repository <i> (e.g., run_all_timepoints.sh) </i>

The output of the model is compared against biological kinematic data to ensure, we are reproducing biologically observed deficits. The github page used to analyse the kinematic data is linked below. 

https://github.com/Allodi-Lab/AngusGray-V0d-Biological-scripts
