# Zig Zag 

## Overview
Simulation of event chain MCMC processes.

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
simulator = MetropolisSimulator

[Simulator]
target = StandardGaussTarget
num_samples = 30000
x0 = 0.0

[SimulatorSpecific]
noise_distribution = GaussianNoiseDistribution
sigma_noise = 2.38

[Target]
dim = 1
mu_target = 0.0
sigma_target = 1.0

[Analysis]
do_timestamp = False
do_plot_samples = True
do_compare_cdf = False
do_plot_zigzag = False

do_reference = False 
reference_simulator = Metropolis 

do_autocorr = False
max_lag = 50
do_write_autocorr_samples = False
do_plot_autocorr = False
do_compare_autocorr = False

do_mean_squared_displacement = False
```
## Run
```bash
python main.py --config config_files/config_filename.ini
```

