import os 
import configparser
from datetime import datetime
from utils.sim_utils import print_section
from utils.build_utils import parse_value, parse_possible_list
from targets.standard_gauss_target import StandardGaussTarget
from simulators.event_chain_simulator import EventChainSimulator
from simulators.metropolis_simulator import MetropolisSimulator
import analysis as anl

def main(config_file):
    """
    Main function runs workflow based on config file 'config.json'.

    Parameters
    ---
    config_file : str
        Path to config file containing simulation parameters and options.

    Workflow
    ---
        - Initialises selected models with provided parameters.
        - Creates directory for saving simulation results and parameters.
        - Runs simulation and selected analysis.
    """
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_file)
    # Run 
    simulator_name = config.get("Run", "simulator")
    # Simulator 
    target_name = config.get("Simulator", "target")
    num_samples = config.getint("Simulator", "num_samples")
    x0 = parse_possible_list(config.get("Simulator", "x0"))
    # Specific to chosen simulator 
    simulator_specific_params = {key: parse_value(value) for key, value in config.items("SimulatorSpecific")}
    # Target
    dim = config.getint("Target", "dim")
    target_params = {key: parse_value(value) for key, value in config.items("Target")}
    # Analysis
    do_timestamp = config.getboolean("Analysis", "do_timestamp")
    do_plot_samples = config.getboolean("Analysis", "do_plot_samples")
    do_compare_cdf = config.getboolean("Analysis", "do_compare_cdf")
    do_plot_zigzag = config.getboolean("Analysis", "do_plot_zigzag")

    do_reference = config.getboolean("Analysis", "do_reference")
    reference_simulator = config.get("Analysis", "reference_simulator")

    do_autocorr = config.getboolean("Analysis", "do_autocorr")
    max_lag = config.getint("Analysis", "max_lag")
    do_write_autocorr_samples = config.getboolean("Analysis", "do_write_autocorr_samples")
    do_plot_autocorr = config.getboolean("Analysis", "do_plot_autocorr")
    do_compare_autocorr = config.getboolean("Analysis", "do_compare_autocorr")

    do_mean_squared_displacement = config.getboolean("Analysis", "do_mean_squared_displacement")
    # Create output directory
    if do_timestamp:
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        output_dir = os.path.join('data', target_name, simulator_name, f'_num_samples={num_samples}_x0={x0}', timestamp)
    else:
        output_dir = os.path.join('data', target_name, simulator_name, f'_num_samples={num_samples}_x0={x0}')
    os.makedirs(output_dir, exist_ok=True)

    print_section("Beginning Workflow")

    # Build target
    target_class = globals().get(target_name)
    if target_class is None:
        raise ValueError(f"Target class '{target_name}' not found.")
    target = target_class(dim=dim, target_params=target_params)
    print(f'\n{target_name} target initalised.')
    
    # Build simulator
    simulator_class = globals().get(simulator_name)
    if simulator_class is None:
        raise ValueError(f'Simulator class {simulator_name} not found')
    simulator = simulator_class(target=target, num_samples=num_samples, x0=x0, simulator_specific_params=simulator_specific_params)
    print(f'\n{simulator_name} simulator initalised.')
    simulator.sim(output_dir)
    
    # Analysis
    if do_plot_samples:
        if dim == 1:
            anl.plot_samples1d(output_dir, target_name, simulator_name)
        elif dim == 2:
            anl.plot_samples2d(output_dir, target_name, simulator_name)
    if do_plot_zigzag:
        anl.plot_zigzag(output_dir, target_name)
    if do_autocorr:
        anl.autocorr(output_dir, max_lag, target_name, do_write_autocorr_samples, do_plot_autocorr)
    if do_mean_squared_displacement:
        anl.mean_squared_displacement(output_dir, target_name)

    # Joint analysis
    #if do_compare_cdf:
    #    anl.compare_cdf(output_dir, method1, method2)
    #if do_compare_autocorr:
    #    anl.compare_autocorr(output_dir, max_lag, method1, method2)

    print_section("Workflow Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run workflow based on configuration file.")
    parser.add_argument('--config', type=str, default='config_files/metropolis.ini', help="Path to the JSON configuration file.")
    args = parser.parse_args()
    main(args.config)
