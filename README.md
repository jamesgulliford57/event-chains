# Zigzag 

## Overview
Simulation of event chain MCMC processes. This project is Work in Progress.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jamesgulliford57/zigzag.git
cd zigzag
pip install -r requirements.txt
```

## Configuration 
Configuration files can be found in /config_files.
### Example Configuration
```bash
[Run]
simulator = EventChainSimulator

[Simulator]
do_simulation = True
target = GaussTarget
num_samples = 30000
x0 = 0.0

[SimulatorSpecific]
v0 = 1.0
final_time = 30000.0
poisson_thinned = False

[Target]
mu_target = 0.0
sigma_target = 1.0

[ReferenceSimulator]
do_reference_simulation = True 
reference_simulator = MetropolisSimulator 

[ReferenceSimulatorSpecific]
noise_distribution = GaussianNoiseDistribution
sigma_noise = 2.38

[Analysis]
do_timestamp = False
do_plot_samples = True

do_plot_zigzag = True
normalise_zigzag = True

do_compare_cdf = True
do_norm_compare_cdf = True

do_autocorr = True
max_lag = 50
autocorr_method = angular
do_write_autocorr_samples = True
do_plot_autocorr = True
do_compare_autocorr = True

do_mean_squared_displacement = True
```
## Run
```bash
python main.py --config config_files/config_filename.ini
```

