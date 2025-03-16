# Zig Zag 

## Overview
Simulation of event chain MCMC processes. This project is Work in Progress.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jamesgulliford57/zigzag.git
cd zigzag
pip install -r requirements.txt (requirements tbc)
```

## Configuration 
Configuration can be set within the config.ini file. 
### Example Configuration
```bash
[Run]
simulator = EventChainSimulator

[Simulator]
do_simulation = True
target = GaussTarget
num_samples = 30000 
x0 = [0.0, 0.0]

[SimulatorSpecific]
v0 = [1.0, 1.0]
final_time = 30000.0
poisson_thinned = True

[Target]
mu_target = 0.0
sigma_target = 1.0

[ReferenceSimulator]
do_reference_simulation = False 
reference_simulator = MetropolisSimulator 

[ReferenceSimulatorSpecific]
noise_distribution = GaussianNoiseDistribution
sigma_noise = 2.38

[Analysis]
do_timestamp = False
do_plot_samples = True
do_compare_cdf = True
do_plot_zigzag = True

do_autocorr = False
max_lag = 50
do_write_autocorr_samples = True
do_plot_autocorr = False
do_compare_autocorr = True

do_mean_squared_displacement = True
```
## Run
```bash
python main.py --config config_files/config_filename.ini
```

