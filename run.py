import os 
import configparser
from datetime import datetime
from utils.sim_utils import print_section
from utils.build_utils import parse_value, parse_possible_list, list_files_excluding
from targets.gauss_target import GaussTarget
from targets.boltzmann_target import BoltzmannTarget
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
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found. Available: {os.listdir('config_files')}")
    # Load configuration 
    config = configparser.ConfigParser()
    config.read(config_file)
    # Run 
    simulator_name = config.get("Run", "simulator")
    do_simulation = config.getboolean("Run", "do_simulation")
    # Simulator 
    target_name = config.get("Simulator", "target")
    try:
        num_samples = config.getint("Simulator", "num_samples")
    except:
       raise ValueError(f'Number of samples must be an integer. Provided: {config.get("Simulator", "num_samples")}')
    x0 = parse_possible_list(config.get("Simulator", "x0"))
    dim = len(x0)
    # Specific to chosen simulator 
    simulator_specific_params = {key: parse_value(value) for key, value in config.items("SimulatorSpecificParams")}
    # Target
    target_params = {'dim' : dim} | {key: parse_value(value) for key, value in config.items("TargetParams")}
    # Sampler
    samplers = parse_possible_list(config.get("Sampler", "samplers"))
    # Reference
    do_reference_simulation = config.getboolean("ReferenceSimulator", "do_reference_simulation")
    reference_simulator_name = config.get("ReferenceSimulator", "reference_simulator")
    if do_reference_simulation:
        reference_simulator_specific_params = {key: parse_value(value) for key, value in config.items("ReferenceSimulatorSpecific")}
    # Analysis
    do_timestamp = config.getboolean("Analysis", "do_timestamp")
    do_plot_samples = config.getboolean("Analysis", "do_plot_samples")

    do_compare_cdf = config.getboolean("Analysis", "do_compare_cdf")
    do_norm_compare_cdf = config.getboolean("Analysis", "do_norm_compare_cdf")
    do_cramer_von_mises = config.getboolean("Analysis", "do_cramer_von_mises")

    do_autocorr = config.getboolean("Analysis", "do_autocorr")
    max_lag = config.getint("Analysis", "max_lag")
    autocorr_method = config.get("Analysis", "autocorr_method")
    autocorr_samples = config.get("Analysis", "autocorr_samples")    
    do_write_autocorr_samples = config.getboolean("Analysis", "do_write_autocorr_samples")
    do_plot_autocorr = config.getboolean("Analysis", "do_plot_autocorr")
    do_compare_autocorr = config.getboolean("Analysis", "do_compare_autocorr")

    do_mean_squared_displacement = config.getboolean("Analysis", "do_mean_squared_displacement")
    # Create output directory
    if do_timestamp:
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        directory = os.path.join('data', target_name, f'{"".join(f"{key}={value}_" for key, value in target_params.items())}', simulator_name, f'num_samples={num_samples}_x0={x0}', timestamp)
    else:
        directory = os.path.join('data', target_name, f'{"".join(f"{key}={value}_" for key, value in target_params.items())}', simulator_name, f'num_samples={num_samples}_x0={x0}')
    os.makedirs(directory, exist_ok=True)
    if do_reference_simulation:
            reference_directory = os.path.join(directory, 'reference')

    print_section("Beginning Workflow")

    # Build target
    if do_simulation:
        target_class = globals().get(target_name)
        if target_class is None:
            raise ValueError(f"Target class '{target_name}' not found. Available: {list_files_excluding('targets', 'target.py')}")
        target = target_class(dim=dim, target_params=target_params)
        print(f'\n{target_name} target with parameters {target_params} initalised.')
    
        # Build simulator
        simulator_class = globals().get(simulator_name)
        if simulator_class is None:
            raise ValueError(f'Simulator class {simulator_name} not found. Available: {list_files_excluding("simulators", "simulator.py")}')
        simulator = simulator_class(target=target, num_samples=num_samples, x0=x0, samplers=samplers, simulator_specific_params=simulator_specific_params)
        print(f'\n{simulator_name} simulator with parameters {simulator_specific_params} initialised.')
        
        # Build reference simulator
        if do_reference_simulation:
            reference_simulator_class = globals().get(reference_simulator_name)
            if reference_simulator_class is None:
                raise ValueError(f'Reference simulator class {reference_simulator_name} not found. Available: {list_files_excluding("simulators", "simulator.py")}')
            reference_simulator = reference_simulator_class(target=target, num_samples=num_samples, x0=x0, samplers=samplers, simulator_specific_params=reference_simulator_specific_params)
            print(f'\n{reference_simulator_name} reference simulator with parameters {reference_simulator_specific_params} initialised.')

        # Simulation
        simulator.sim(directory=directory)
        if do_reference_simulation:
            os.makedirs(reference_directory, exist_ok=True)
            reference_simulator.sim(directory=reference_directory)
    
    # Analysis
    if do_plot_samples:
        anl.plot_samples(directory=directory)
        if do_reference_simulation:
            anl.plot_samples(directory=reference_directory)
    if do_autocorr:
        anl.autocorr(directory=directory, max_lag=max_lag, autocorr_method=autocorr_method, autocorr_samples=autocorr_samples, do_write_autocorr_samples=do_write_autocorr_samples, do_plot_autocorr=do_plot_autocorr)
        if do_reference_simulation:
            anl.autocorr(directory=reference_directory, max_lag=max_lag, autocorr_method=autocorr_method, autocorr_samples=autocorr_samples, do_write_autocorr_samples=do_write_autocorr_samples, do_plot_autocorr=do_plot_autocorr)
    if do_mean_squared_displacement:
        anl.mean_squared_displacement(directory=directory)
        if do_reference_simulation:
            anl.mean_squared_displacement(directory=reference_directory)
    if do_cramer_von_mises:
        anl.cramer_von_mises(directory=directory)
        #if do_reference_simulation:
        #    anl.cramer_von_mises(directory=reference_directory)

    # Joint analysis
    if do_compare_cdf:
            anl.compare_cdf(directory=directory)
    if do_norm_compare_cdf:
            anl.compare_norm_cdf(directory=directory)
    if do_compare_autocorr:
        anl.compare_autocorr(directory=directory, max_lag=max_lag, autocorr_method=autocorr_method, do_write_autocorr_samples=do_write_autocorr_samples)

    print_section("Workflow Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run workflow based on configuration file.")
    parser.add_argument('--config', type=str, default='config_files/metropolis.ini', help="Path to the JSON configuration file.")
    args = parser.parse_args()
    main(args.config)
