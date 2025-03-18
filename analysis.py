import os
import numpy as np
import matplotlib.pyplot as plt
import json
from utils.data_utils import write_npy, read_json, set_colors

plt.style.use('ggplot')
plt.rcParams['agg.path.chunksize'] = 10000

def plot_samples(directory, initial_samples=500, figsize=(16, 20)):
    """
    Produces plot with 4 subplots: i) Samples, ii) First 500 samples,
    iii) Empirical PDF, iv) Empirical CDF
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    initial_samples: int
        Number of initial samples to plot in second subplot.
    figsize: tuple
        Size of figure.
    """
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    output_path = os.path.join(directory, f"output.json")
    # Load files
    samples = np.load(samples_path)
    samples = np.atleast_2d(samples)
    output = read_json(output_path)
    # Load variables from output dict
    target_name = output['target_name']
    simulator_name = output['simulator_name']

    print(f"\nPlotting samples for {target_name} {simulator_name} simulation...")
    # Number of model components 
    dim = np.shape(samples)[0]
    # Format colours
    cpt_colors = set_colors(dim)
    # Create figure and plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    for cpt in range(dim):
        # Plot all samples
        ax1.plot(samples[cpt, :], color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}]', linewidth=1)
        # Plot first {initial_samples} samples
        ax2.plot(samples[cpt, :][:initial_samples], color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}][0:{initial_samples}]', linewidth=1)
        # Plot empirical PDF
        ax3.hist(samples[cpt, :], bins=50, color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}]')
        # Plot empirical CDF
        cdf_values = []
        sorted_samples = np.sort(samples[cpt, :])
        cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax4.plot(sorted_samples, cdf_values, color=cpt_colors[cpt], alpha=0.6, label=f'Samples[{cpt}]', linewidth=1)
    # Create legends 
    if dim < 5:
        ax1.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
        ax2.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
        ax3.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
        ax4.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=14)
    
    fig.suptitle(f'{target_name} {simulator_name} {np.shape(samples)[1]} samples', fontsize=32, y=0.93)
    
    # Save output plot
    output_file = os.path.join(directory, f"samples_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"{target_name} {simulator_name} samples plot saved to {output_file}")

    plt.close()

def plot_zigzag(directory, num_events=200, normalised=False, figsize=(10, 8)):
    """
    Plots zigzag trajectory. Trajectory visualises events and velocity flips.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    num_events: int
        Number of events to plot.
    normalised: bool
        Normalise data for compactness.
    figsize: tuple
        Size of figure.
    """
    # Look for event states
    try:
        event_states_path = os.path.join(directory, f"event_states.npy")
    except:
        raise FileNotFoundError(f"{event_states_path} not found")
    event_states = np.load(event_states_path)
    # Identify file paths
    output_path = os.path.join(directory, 'output.json')
    # Load files
    output = read_json(output_path)
    # Load variables from output dict
    target_name = output['target_name']
    simulator_name = output['simulator_name']

    print(f"\nPlotting zigzag for {target_name} {simulator_name} simulation...")

    x = event_states[0, :int(num_events)] 
    y = event_states[1, :int(num_events)]
    # Normalise for compactness
    if normalised:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x,y)
    # Add start and end points
    ax.scatter(x[0], y[0], color='green', s=100, label='Start',
              zorder=5, edgecolor='white', linewidth=2)
    ax.scatter(x[-1], y[-1], color='red', s=100, label='End',
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

def compare_cdf(directory):
    """
    Compare CDFs of simulation samples from two different methods to test for 
    convergence to same stationary distribution.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    """
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    reference_samples_path = os.path.join(directory, f"reference/position_samples.npy")
    output_path = os.path.join(directory, 'output.json')
    reference_output_path = os.path.join(directory, 'reference/output.json')
    # Load files
    samples = np.load(samples_path)
    samples = np.atleast_2d(samples)
    reference_samples = np.load(reference_samples_path)
    reference_samples = np.atleast_2d(reference_samples)
    # Load output
    output = read_json(output_path)
    reference_output = read_json(reference_output_path)

    target_name = output['target_name']
    simulator_name = output['simulator_name']
    reference_simulator_name = reference_output['simulator_name']

    print(f"\nComparing CDFs for {target_name} {simulator_name} and {reference_simulator_name} simulation samples...")

    dim = np.shape(samples)[0]

    cpt_colors = set_colors(dim)
    # Create figure and plot
    fig, ax = plt.subplots()
    for cpt in range(dim):
        # CDF 1
        sorted_samples = np.sort(samples[cpt, :])
        cdf_values1 = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax.plot(sorted_samples, cdf_values1, linestyle = '-', color = cpt_colors[cpt], alpha=0.5, linewidth = 2, label=f'{simulator_name}[{cpt}]')
        # CDF 2
        sorted_reference_samples = np.sort(reference_samples[cpt, :])
        cdf_values2 = np.arange(1, len(sorted_reference_samples) + 1) / len(sorted_reference_samples)
        ax.plot(sorted_reference_samples, cdf_values2, linestyle = '--', color = cpt_colors[cpt], alpha=0.5, label=f'{reference_simulator_name}[{cpt}]')
    
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

def autocorr(directory, max_lag=50, autocorr_method='component', do_write_autocorr_samples=False, do_plot_autocorr=False):
    """
    Returns the autocorrelation of simulation samples as a function of lag up to
    the provided max_lag. Also returns the integrated autocorrelation function (IAT)
    and effective sample size.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    max_lag: int
        Maximum lag for autocorrelation calculation.
    autocorr_method: str
        Method for calculating autocorrelation. 
        Options: 'component' : Calculate autocorrelation for each component separately.
                 'vector' : Calculate autocorrelation for the vector norm of the samples.
    do_write_autocorr_samples: bool
        Write autocorrelation samples to file.
    do_plot_autocorr: bool
        Plot autocorrelation samples.
    """
    max_lag = int(max_lag)
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    output_path = os.path.join(directory, f"output.json")
    # Load files
    samples = np.load(samples_path)
    samples = np.atleast_2d(samples)
    output = read_json(output_path)

    target_name = output['target_name']
    simulator_name = output['simulator_name']

    dim = np.shape(samples)[0]
    # Calculate autocorrelation
    if autocorr_method == 'component':
        autocorr_samples = np.zeros((dim, max_lag + 1))
        iat = [] 
        eff_sample_size = []
        for cpt in range(dim):
            mean = np.mean(samples[cpt])
            var = np.var(samples[cpt])
            autocorr_samples_cpt = [(np.mean(samples[cpt][k:] * samples[cpt][:-k]) - mean ** 2) / var for k in range(1, max_lag + 1)]
            autocorr_samples_cpt = np.insert(autocorr_samples_cpt, 0, 1)
            autocorr_samples[cpt, :] = autocorr_samples_cpt
            # Calculation integrated autocorrelation time
            iat_cpt = 1 + 2 * np.sum(autocorr_samples_cpt)
            iat.append(iat_cpt)
            eff_sample_size_cpt = len(samples[cpt, :]) / iat_cpt 
            eff_sample_size.append(eff_sample_size_cpt)
    
    elif autocorr_method == 'vector':
        autocorr_samples = np.array([np.mean(np.sum(samples[:, k:] * samples[:, :-k], axis=0)) / np.mean(np.sum(samples**2, axis=0)) for k in range(1, max_lag + 1)])
        autocorr_samples = np.insert(autocorr_samples, 0, 1, axis=0)
        iat = 1 + 2 * np.sum(autocorr_samples, axis=0)
        eff_sample_size = len(samples[0, :]) / iat
        iat = [iat]
        eff_sample_size = [eff_sample_size]

    elif autocorr_method == 'angular':
        norms = np.linalg.norm(samples, axis=0, keepdims=True)
        unit_vectors = samples / (norms + 1e-6)
        autocorr_samples = np.array([np.mean(np.sum(unit_vectors[:, k:] * unit_vectors[:, :-k], axis=0)) for k in range(1, max_lag + 1)])
        autocorr_samples = np.insert(autocorr_samples, 0, 1, axis=0)
        iat = 1 + 2 * np.sum(autocorr_samples, axis=0)
        eff_sample_size = len(samples[0, :]) / iat
        iat = [iat]
        eff_sample_size = [eff_sample_size]

    autocorr_samples = np.atleast_2d(autocorr_samples)
    # Write autocorrelation samples
    if do_write_autocorr_samples:
        write_npy(directory, **{f"autocorr_samples": autocorr_samples})
    
    # Plot autocorrelation samples
    if do_plot_autocorr:
        fig, ax = plt.subplots()
        cpt_colors = set_colors(dim)
        if autocorr_method == 'component':
            for cpt in range(dim):
                ax.plot(autocorr_samples[cpt, :], linestyle='-', color=cpt_colors[cpt], alpha=0.5, label=f'{simulator_name}[{cpt}]:\nIAT = {iat[cpt]:.2f}, N_eff = {eff_sample_size[cpt]:.0f}')
        elif autocorr_method == 'vector' or autocorr_method == 'angular':
            ax.plot(autocorr_samples[0, :], linestyle='-', color=cpt_colors[0], alpha=0.5, label=f'{simulator_name}:\nIAT = {iat[0]:.2f}, N_eff = {eff_sample_size[0]:.0f}')
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        fig.suptitle(f'{target_name} {autocorr_method.capitalize()} Autocorrelation Comparison', fontsize=14, y=0.95)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)

        output_file = os.path.join(directory, f"autocorr_plot.png")
        plt.savefig(output_file, dpi=400)
        print(f"{target_name} {simulator_name} samples autocorrelation plot saved to {output_file}")
    
    # Write IAT and effective sample size
    output['iat'] = iat 
    output['eff_sample_size'] = eff_sample_size
    with open(output_path, 'w') as f:
         json.dump(output, f, indent=4)

    return autocorr_samples, iat, eff_sample_size, dim

def compare_autocorr(directory, max_lag=50, autocorr_method='component', do_write_autocorr_samples=False):
    """
    Plots autocorrelation samples as a function of lag for two provided methods for
    comparison. 
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    max_lag: int
        Maximum lag for autocorrelation calculation.
    autocorr_method: str
        Method for calculating autocorrelation.
    do_write_autocorr_samples: bool
        Write autocorrelation samples to file.
    """
    output_path = os.path.join(directory, f"output.json")
    reference_output_path = os.path.join(directory, 'reference/output.json')
    output = read_json(output_path)
    reference_output = read_json(reference_output_path)

    target_name = output['target_name']
    simulator_name = output['simulator_name']
    reference_simulator_name = reference_output['simulator_name']
    
    print(f"\nComparing correlation functions for {target_name} {simulator_name} and {reference_simulator_name} simulations...")

    reference_directory = os.path.join(directory, 'reference')
    autocorr1, iat1, eff_sample_size1, dim = autocorr(directory, max_lag, autocorr_method, do_write_autocorr_samples)
    autocorr2, iat2, eff_sample_size2, _ = autocorr(reference_directory, max_lag, autocorr_method, do_write_autocorr_samples) 
    # Plot
    fig, ax = plt.subplots()
    cpt_colors = set_colors(dim)
    if autocorr_method == 'component':
        for cpt in range(dim):
            ax.plot(autocorr1[cpt, :], linestyle='-', color=cpt_colors[cpt], alpha=0.5, label=f'{simulator_name}[{cpt}]:\nIAT = {iat1[cpt]:.2f}, N_eff = {eff_sample_size1[cpt]:.0f}')
            ax.plot(autocorr2[cpt, :], linestyle='--', color=cpt_colors[cpt], alpha=0.5, label=f'{reference_simulator_name}[{cpt}]:\nIAT = {iat2[cpt]:.2f}, N_eff = {eff_sample_size2[cpt]:.0f}')
    elif autocorr_method == 'vector' or autocorr_method == 'angular':
        ax.plot(autocorr1[0, :], linestyle='-', color=cpt_colors[0], alpha=0.5, label=f'{simulator_name}:\nIAT = {iat1[0]:.2f}, N_eff = {eff_sample_size1[0]:.0f}')
        ax.plot(autocorr2[0, :], linestyle='--', color=cpt_colors[0], alpha=0.5, label=f'{reference_simulator_name}:\nIAT = {iat2[0]:.2f}, N_eff = {eff_sample_size2[0]:.0f}')
    

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    fig.suptitle(f'{target_name} {autocorr_method.capitalize()} Autocorrelation Comparison', fontsize=14, y=0.95)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    # Save files
    output_file = os.path.join(directory, f"compare_autocorr_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"{target_name} {simulator_name}, {reference_simulator_name} autocorrelation comparison plot saved to {output_file}")

    plt.close()

def mean_squared_displacement(directory):
    """
    Returns mean squared displacement of samples.
    
    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    """
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    output_path = os.path.join(directory, 'output.json')
    # Load files
    samples = np.load(samples_path)
    output = read_json(output_path)

    target_name = output['target_name']
    simulator_name = output['simulator_name']

    print(f"Calculating mean squared displacement for {target_name} {simulator_name} samples...")    

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

def compare_norm_cdf(directory):
    """
    Compare CDFs of norm of 2D simulation samples from two different methods to test for
    convergence to same stationary distribution.

    Parameters
    ---
    directory: str
        Path to directory containing simulation files.
    """
    # Identify file paths
    samples_path = os.path.join(directory, f"position_samples.npy")
    reference_samples_path = os.path.join(directory, f"reference/position_samples.npy")
    output_path = os.path.join(directory, 'output.json')
    reference_output_path = os.path.join(directory, 'reference/output.json')
    # Load files
    samples = np.load(samples_path)
    samples = np.atleast_2d(samples)
    reference_samples = np.load(reference_samples_path)
    reference_samples = np.atleast_2d(reference_samples)
    output = read_json(output_path)
    reference_output = read_json(reference_output_path)
    dim = np.shape(samples)[0]

    target_name = output['target_name']
    simulator_name = output['simulator_name']
    reference_simulator_name = reference_output['simulator_name']

    print(f"\nComparing norm CDFs for {target_name} {simulator_name} and {reference_simulator_name} simulation samples...")
    # Calculate norm
    samples = np.array(sum(samples[i, :]**2 for i in range(dim)))**0.5
    reference_samples = np.array(sum(reference_samples[i, :]**2 for i in range(dim)))**0.5

    fig, ax = plt.subplots()
    # Create cdf 1
    sorted_samples = np.sort(samples)
    cdf_values1 = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax.plot(sorted_samples, cdf_values1, linestyle = '-', color = 'r', alpha=0.5, linewidth = 2, label = simulator_name)
    # Create cdf 2
    sorted_reference_samples = np.sort(reference_samples)
    cdf_values2 = np.arange(1, len(sorted_reference_samples) + 1) / len(sorted_reference_samples)
    ax.plot(sorted_reference_samples, cdf_values2, linestyle = '--', color = 'k', alpha=0.5, label = reference_simulator_name)
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

