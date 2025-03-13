import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from utils.data_utils import write_npy

plt.style.use('ggplot')
plt.rcParams['agg.path.chunksize'] = 10000

def plot_samples(directory, figsize=(16, 20)):
    """
    Produces plot with 4 subplots: i) Samples, ii) First 500 samples,
    iii) Empirical PDF, iv) Empirical CDF
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    figsize: tuple
        Size of figure.
    """
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    output_path = os.path.join(directory, f"output.json")
    # Load files
    samples = np.atleast_2d(np.load(samples_path))
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                output = json.load(f)
        except json.JSONDecodeError:
            output = {}
    else:
        output = {}

    target_name = output['target_name']
    simulator_name = output['simulator_name']

    print(f"\nPlotting samples for {target_name} {simulator_name} simulation...")

    dim = np.shape(samples)[0]

    colors = ["firebrick", "black", "dimgray", "darkred", "brown", "maroon", "gray", "darkslategray"]
    n_colors = len(colors)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_colors)
    if dim == 1:
        cpt_colors = ["firebrick"]  
    else:
        cpt_colors = [custom_cmap(i / (dim - 1)) for i in range(dim)]
    # Create figure and plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    for cpt in range(dim):
        ax1.plot(samples[cpt, :], color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}]', linewidth=1)
    if dim < 5:
        ax1.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
    
    for cpt in range(dim):
        ax2.plot(samples[cpt, :][:500], color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}][0:500]', linewidth=1)
    if dim < 5:
        ax2.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
    
    for cpt in range(dim):
        ax3.hist(samples[cpt, :], bins=50, color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}]')
    if dim < 5:
        ax3.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
    
    for cpt in range(dim):
        cdf_values = []
        sorted_samples = np.sort(samples[cpt, :])
        cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax4.plot(sorted_samples, cdf_values, color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}]', linewidth=1)
    if dim < 5:
        ax4.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
    
    fig.suptitle(f'{target_name} {simulator_name} {np.shape(samples)[1]} samples', fontsize=32, y=0.93)
    
    # Save output to file
    output_file = os.path.join(directory, f"samples_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"{target_name} {simulator_name} samples plot saved to {output_file}")

    plt.close()

def plot_zigzag(directory, target_name, simulator_name, num_events=200, normalised=False, figsize=(10, 8)):
    """
    Plots zigzag trajectory from initial value to end point defined by num_points. 
    Trajectory visualises events and velocity flips.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    method: str
        Simulation method name 
    normalised: bool
        Normalise data to remove gaps
    figsize: tuple
        Size of figure
    num_points: int 
        Number of events to display on trajectory
    """
    print(f"\nPlotting zigzag for {target_name} {simulator_name} simulation...")

    fig, ax = plt.subplots(figsize=figsize)

    try:
        event_states_path = os.path.join(directory, f"event_states.npy")
    except:
        raise FileNotFoundError(f"{event_states_path} not found")
    event_states = np.load(event_states_path)

    json_path = os.path.join(directory, 'output.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
                data = {}
    try:
        thinned_acceptance_rate = data['thinned_acceptance_rate']  
    except:
        thinned_acceptance_rate = 1
    # Scale data to remove gaps
    x = event_states[0, :int(num_events / thinned_acceptance_rate)] 
    y = event_states[1, :int(num_events / thinned_acceptance_rate)]
    # Normalise the data to remove gaps
    if normalised:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
    # Plot segments with gradient color using normalized data
    ax.plot(x,y)
    # Add start and end points on normalized data
    ax.scatter(x[0], y[0], 
              color='green', s=100, label='Start',
              zorder=5, edgecolor='white', linewidth=2)
    ax.scatter(x[-1], y[-1], 
              color='red', s=100, label='End',
              zorder=5, edgecolor='white', linewidth=2)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Component 0', fontsize=12)
    ax.set_ylabel('Component 1', fontsize=12)
    plt.title('Zig Zag Sampler Trajectory', fontsize=14, pad=20)
    
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    plt.tight_layout()
    
    output_file = os.path.join(directory, f"zigzag_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"{target_name} {simulator_name} samples plot saved to {output_file}")
    
    plt.close()

def compare_cdf(directory, target_name, simulator_name, reference_simulator_name):
    """
    Compare CDFs of simulation samples from two different methods to test for 
    convergence to same stationary distribution.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    method1: str
        Simulation method1 name 
    method2: str        
        Simulation method2 name
    """
    print(f"\nComparing CDFs for {target_name} {simulator_name} and {reference_simulator_name} simulation samples...")

    # Identify file paths
    samples1_path = os.path.join(directory, f"position_samples.npy")
    samples2_path = os.path.join(directory, f"reference/position_samples.npy")
    # Load files
    samples1 = np.load(samples1_path)
    samples1 = np.atleast_2d(samples1)
    samples2 = np.load(samples2_path)
    samples2 = np.atleast_2d(samples2)

    dim = np.shape(samples1)[0]

    colors = ["firebrick", "black", "dimgray", "darkred", "brown", "maroon", "gray", "darkslategray"]
    n_colors = len(colors)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_colors)
    if dim == 1:
        cpt_colors = ["firebrick"]  
    else:
        cpt_colors = [custom_cmap(i / (dim - 1)) for i in range(dim)]
    # Create figure and plot
    fig, ax = plt.subplots()
    # Create cdf 1
    for cpt in range(dim):
        sorted_samples1 = np.sort(samples1[cpt, :])
        cdf_values1 = np.arange(1, len(sorted_samples1) + 1) / len(sorted_samples1)
        ax.plot(sorted_samples1, cdf_values1, linestyle = '-', color = cpt_colors[cpt], alpha=0.5, linewidth = 2, label=f'{simulator_name}[{cpt}]')
    # Create cdf 2
    for cpt in range(dim):
        sorted_samples2 = np.sort(samples2[cpt, :])
        cdf_values2 = np.arange(1, len(sorted_samples2) + 1) / len(sorted_samples2)
        ax.plot(sorted_samples2, cdf_values2, linestyle = '--', color = cpt_colors[cpt], alpha=0.5, label=f'{reference_simulator_name}[{cpt}]')
    
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
    ax.set_xlabel('x', fontsize=12, color='black')
    ax.set_ylabel('CDF', fontsize=12, color='black')
    fig.suptitle(f'{target_name} CDF Comparison', fontsize=14, color='black', y=0.95)
    if dim < 5:
        ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)

    output_file = os.path.join(directory, f"cdf_compare_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"Compare CDF plot saved to {output_file}")

    plt.close()
     
def pdmp_ks_test(samples, dist, method):
    """
    Performs Kolmogorov-Smirnov test on simulation samples to test for convergence
    to the provided distribution.
    
    Parameters
    ---
    samples: np.array
        Samples to test.
    dist: str
        Distribution to test against e.g. 'norm', 'expon', 'uniform'. 
    method: str
        Simulation method name.
    """    
    print(f"\nPerforming KS test for {method}...")

    from scipy.stats import kstest
    _, p = kstest(samples, dist, args=(np.mean(samples), np.std(samples)))
    if p > 0.05:
        print(f"{method} samples are likely normally distributed, p = {p}")
    else:
        print(f"{method} Samples are likely not normally distributed, p = {p}")

def autocorr(directory, max_lag, target_name, simulator_name, do_write_autocorr_samples=False, do_plot_autocorr=False):
    """
    Returns the autocorrelation of simulation samples as a function of lag up to
    the provided max_lag. Also returns the integrated autocorrelation function (IAT)
    and effective sample size.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    max_lag: int
        Maximum lag for autocorrelation calculation
    method: str
        Simulation method name
    do_write_autocorr_samples: bool
        Write autocorrelation samples to file
    do_plot_autocorr: bool
        Plot autocorrelation samples
    """
    print(f"\nCalculating autocorrelation for {target_name} {simulator_name} simulation samples...")

    max_lag = int(max_lag)
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    # Load files
    samples = np.load(samples_path).flatten()
    mean = np.mean(samples)
    var = np.var(samples)
    autocorr_samples = [(np.mean(samples[k:] * samples[:-k]) - mean ** 2) / var for k in range(1, max_lag + 1)]
    autocorr_samples = np.insert(autocorr_samples, 0, 1)
    
    # Calculation integrated autocorrelation time
    iat = 1 + 2 * np.sum(autocorr_samples)
    eff_sample_size = len(samples) / iat
    # Write autocorrelation samples
    if do_write_autocorr_samples:
        write_npy(directory, **{f"autocorr_samples": autocorr_samples})
    
    # Plot autocorrelation samples
    if do_plot_autocorr:
        fig, ax = plt.subplots()
        ax.plot(autocorr_samples)
        output_file = os.path.join(directory, f"autocorr_plot.png")
        plt.savefig(output_file, dpi=400)
        print(f"{target_name} {simulator_name} samples autocorrelation plot saved to {output_file}")
    
    # Read existing parameters JSON
    json_path = os.path.join(directory, f"output.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    
    # Write IAT to JSON
    data['iat'] = iat 
    data['eff_sample_size'] = eff_sample_size
    with open(json_path, 'w') as f:
         json.dump(data, f, indent=4)

    return autocorr_samples, iat, eff_sample_size

def compare_autocorr(directory, max_lag, target_name, simulator_name, reference_simulator_name, do_write_autocorr_samples=False):
    """
    Plots autocorrelation samples as a function of lag for two provided methods for
    comparison. 
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    max_lag: int
        Maximum lag for autocorrelation calculation
    method1: str
        Simulation method1 name
    method2: str
        Simulation method2 name
    do_write_autocorr_samples: bool
        Write autocorrelation samples to file
    do_compare_autocorr: bool
        Compare autocorrelation samples from different methods
    """
    print(f"\nComparing correlation functions for {target_name} {simulator_name} and {reference_simulator_name} simulations...")

    reference_directory = os.path.join(directory, 'reference')
    autocorr1, iat1, N_eff1 = autocorr(directory, max_lag, target_name, simulator_name, do_write_autocorr_samples)
    autocorr2, iat2, N_eff2 = autocorr(reference_directory, max_lag, target_name, reference_simulator_name, do_write_autocorr_samples) 
    # Plot
    fig, ax = plt.subplots()
    ax.plot(autocorr1, linestyle = '-', color = 'r', alpha=0.5, label = f'{simulator_name}:\nIAT = {iat1:.2f}, N_eff = {N_eff1:.0f}')
    ax.plot(autocorr2, linestyle = '--', color = 'k', alpha=0.5, label = f'{reference_simulator_name}:\nIAT = {iat2:.2f}, N_eff = {N_eff2:.0f}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    fig.suptitle(f'{target_name} Autocorrelation Comparison', fontsize=14, y=0.95)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    # Save files
    output_file = os.path.join(directory, f"compare_autocorr_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"{target_name} {simulator_name}, {reference_simulator_name} autocorrelation comparison plot saved to {output_file}")

    plt.close()

def mean_squared_displacement(directory, target_name, simulator_name):
    """
    Returns mean squared displacement of samples.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    method: str
        Simulation method name
    mean: float

    """
    print(f"Calculating mean squared displacement for {target_name} {simulator_name} samples...")

    # Identify file path
    samples_path = os.path.join(directory, f"position_samples.npy")
    # Load files
    samples = np.load(samples_path)

    # Calculate mean squared displacement
    mean = np.mean(samples)
    msd = np.mean((samples-mean)**2)

    # Read existing parameters
    json_path = os.path.join(directory, f"output.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    
    # Write MSD to parameters
    data['msd'] = msd
    with open(json_path, 'w') as f:
         json.dump(data, f, indent=4)

    return msd

def compare_norm_cdf(directory, target_name, simulator_name, reference_simulator_name):
    """
    Compare CDFs of norm of 2D simulation samples from two different methods to test for
    convergence to same stationary distribution.

    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    target_name: str
        Name of the target distribution.
    simulator_name: str
        Name of the simulation method.
    reference_simulator_name: str    
        Name of the reference simulation method.
    """
    print(f"\nComparing norm CDFs for {target_name} {simulator_name} and {reference_simulator_name} simulation samples...")

    # Identify file paths
    samples1_path = os.path.join(directory, f"position_samples.npy")
    samples2_path = os.path.join(directory, f"reference/position_samples.npy")
    # Load files
    samples1 = np.load(samples1_path)
    samples1 = np.atleast_2d(samples1)
    samples2 = np.load(samples2_path)
    samples2 = np.atleast_2d(samples2)
    dim = np.shape(samples1)[0]
    # Calculate norm
    samples1 = np.array(sum(samples1[i, :]**2 for i in range(dim)))**0.5
    samples2 = np.array(sum(samples2[i, :]**2 for i in range(dim)))**0.5

    fig, ax = plt.subplots()
    # Create cdf 1
    sorted_samples1 = np.sort(samples1)
    cdf_values1 = np.arange(1, len(sorted_samples1) + 1) / len(sorted_samples1)
    ax.plot(sorted_samples1, cdf_values1, linestyle = '-', color = 'r', alpha=0.5, linewidth = 2, label = simulator_name)
    # Create cdf 2
    sorted_samples2 = np.sort(samples2)
    cdf_values2 = np.arange(1, len(sorted_samples2) + 1) / len(sorted_samples2)
    ax.plot(sorted_samples2, cdf_values2, linestyle = '--', color = 'k', alpha=0.5, label = reference_simulator_name)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
    ax.set_xlabel('Norm', fontsize=12, color='black')
    ax.set_ylabel('CDF', fontsize=12, color='black')
    fig.suptitle(f'{target_name} Norm CDF Comparison', fontsize=14, color='black', y=0.95)
    if dim < 5:
        ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)

    output_file = os.path.join(directory, f"norm_cdf_compare_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"Compare CDF plot saved to {output_file}")

    plt.close()

