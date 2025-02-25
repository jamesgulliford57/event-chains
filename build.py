import os 
import json 
import numpy as np
from datetime import datetime
from utils import print_section
from models.zigzag1d import ZigZag1d, GaussZigZag1d
from models.zigzag2d import ZigZag2d, GaussZigZag2d
from models.random_walk1d import StandardGaussRandomWalk1d
import analysis as anl

def main(config_file):
    # Load configuration from JSON file 
    with open(config_file, 'r') as f:
        config = json.load(f)
    # Simulation parameters
    # Method 1
    method1 = config["method1"]
    x0_1 = config["x0_1"]
    N1 = config["N1"]
    dim1 = config["dim1"]
    # Method 2
    method2 = config["method2"]
    x0_2 = config["x0_2"]
    N2 = config["N2"]
    dim2 = config["dim2"]
    v0 = config["v0"]
    final_time = config["final_time"]
    poisson_thinned = config["poisson_thinned"] # Select whether to use Poisson thinning or analytical event rate
    # Analysis paramters
    max_lag = config["max_lag"]
    output_dir = config["output_dir"]
    do_timestamp = config["do_timestamp"]
    # Select plots and analysis to perform 
    do_plot_samples = config["do_plot_samples"]
    do_plot_zigzag = config["do_plot_zigzag"]
    do_compare_cdf = config["do_compare_cdf"]
    do_autocorr = config["do_autocorr"]
    do_plot_autocorr = config["do_plot_autocorr"]
    do_compare_autocorr = config["do_compare_autocorr"]
    do_write_autocorr_samples = config["do_write_autocorr_samples"]
    do_mean_squared_displacement = config["do_mean_squared_displacement"]
    # Create output directory
    if do_timestamp:
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        output_dir = os.path.join(output_dir, f'{method1}_{method2}', timestamp)
    else:
        output_dir = os.path.join(output_dir, f'{method1}_{method2}')
    os.makedirs(output_dir, exist_ok=True)

    print_section("Beginning Simulation")
    # Method 1
    reference_class = globals().get(method1) # Dynamically instantiate 
    if reference_class is None:
        raise ValueError(f"Method1 class '{method1}' not found.")
    reference = reference_class(N=N1, x0=x0_1)
    reference.sim(output_dir)
    # Analysis method 1
    if do_plot_samples:
        if dim1 == 1:
            anl.plot_samples1d(output_dir, method1)
        elif dim2 == 2:
            anl.plot_samples2d(output_dir, method1)
    if do_autocorr:
        anl.autocorr(output_dir, max_lag, method1, do_write_autocorr_samples, do_plot_autocorr)
    if do_mean_squared_displacement:
        anl.mean_square_displacement(output_dir, method1)
    # Method 2
    pdmp_class = globals().get(method2) # Dynamically instantiate 
    if pdmp_class is None:
        raise ValueError(f"Method2 class '{method2}' not found.")
    pdmp = pdmp_class(N=N2, final_time=final_time, x0=x0_2, v0=v0, dim=dim2, poisson_thinned=poisson_thinned)
    pdmp.sim(output_dir)
    # Analysis method 2
    if do_plot_samples:
        if dim2 == 1:
            anl.plot_samples1d(output_dir, method2)
        elif dim2 == 2:
            anl.plot_samples2d(output_dir, method2)
    if do_plot_zigzag:
        anl.plot_zigzag(output_dir, method2)
    if do_autocorr:
        anl.autocorr(output_dir, max_lag, method2, do_write_autocorr_samples, do_plot_autocorr)
    if do_mean_squared_displacement:
        anl.mean_square_displacement(output_dir, method2)
    # Joint analysis
    if do_compare_cdf:
        anl.compare_cdf(output_dir, method1, method2)
    if do_compare_autocorr:
        anl.compare_autocorr(output_dir, max_lag, method1, method2)


    print_section("Workflow Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run workflow based on configuration file.")
    parser.add_argument('--config', type=str, default='config.json', help="Path to the JSON configuration file.")
    args = parser.parse_args()

    main(args.config)

"""
To do:
- Replace long arguments in classes with *args, **kwargs (learn how to do this well)
- Fix thinned events occured print business
- Print summary table as found in PX912 assignment really detailed output
- Streamline json read/writing into one function
- Shouldn't have to put dimension in explictly, infer it from method selected
- Test on function and class inputs to protect against nonsensical inputs
- Streamline plotting functions
- Good way to determine what final time should be used
- Convert config file to ini if number of controls is going to be too high
- Check norm x^2+y^2 CDF
"""