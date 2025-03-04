# Zig Zag 

## Overview
This package simulates zig zag processes using Piecewise Deterministic Markov Processes (PDMPs) and random walks

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jamesgulliford57/zigzag.git
cd zigzag
pip install -r requirements.txt (requirements tbc)
```

## Configuration 
Configuration can be set within the config.json file. 
### Example Configuration
```bash
{
    "method1" : "StandardGaussRandomWalk1d",
    "x0_1": 0.0,
    "N1": 100000,
    "dim1" : 1,

    "method2" : "GaussZigZag2d",
    "x0_2": [0.0, 0.0],
    "N2" : 100000,
    "v0" : [1, 1],
    "dim2" : 2,
    "final_time" : 100000,
    "poisson_thinned" : false,
    
    "output_dir" : "data",
    "do_timestamp" : false,

    "do_plot_samples" : false,
    "do_compare_cdf" : true,
    "do_plot_zigzag" : true,
    "do_autocorr" : false,
    "max_lag" : 50,
    "do_plot_autocorr" : false,
    "do_compare_autocorr" : true,
    "do_write_autocorr_samples" : false,

    "do_mean_squared_displacement" : true

}
```
## Run
```bash
python main.py --config config.json
```

