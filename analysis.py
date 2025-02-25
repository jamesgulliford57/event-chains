import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import json
from utils import write_npy

def get_directory(file_path):
    return os.path.dirname(file_path)

def plot_samples1d(directory, method):
    # Identify file paths
    samples_path = os.path.join(directory, f"samples_{method}.npy")
    # Load files
    samples = np.load(samples_path)
    if np.shape(samples)[0] != 1:
        raise TypeError(f'Function is for plotting 1d states only. Provided {np.shape(samples)[0]}')
    samples = samples.flatten()
    # Create figure and plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,20))
    ax1.plot(samples)
    ax2.plot(samples[0:500])
    ax3.hist(samples, bins=50)
    cdf_values = []
    sorted_samples = np.sort(samples)
    cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax4.plot(sorted_samples, cdf_values)

    output_file = os.path.join(directory, f"samples_plot_{method}.png")
    plt.savefig(output_file, dpi=400)
    print(f"{method} samples plot saved to {output_file}")

    plt.close()

def plot_samples2d(directory, method):
    # Identify file paths
    samples_path = os.path.join(directory, f"samples_{method}.npy")
    # Load files
    samples = np.load(samples_path)
    if np.shape(samples)[0] != 2:
        raise TypeError(f'Function is for plotting 2d states only. Provided {np.shape(samples)[0]}')
    # Create figure and plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,20))
    ax1.plot(samples[0, :], label="Component 0", alpha=0.5)
    ax2.plot(samples[1, :], label = "Component 1", alpha=0.5)
    ax3.hist(samples[0, :], bins=50, color='b', label="Component 0", alpha=0.5)
    ax3.hist(samples[1, :], bins=50, color = 'r', label = "Component 1", alpha=0.5)
    ax3.legend()
    # Plot cdf
    cdf_values = []
    sorted_samples = np.sort(samples[0, :])
    cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax4.plot(sorted_samples, cdf_values, color='b', label="Component 0")
    cdf_values = []
    sorted_samples = np.sort(samples[1, :])
    cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax4.plot(sorted_samples, cdf_values, color = 'r', label = "Component 1")
    ax4.legend()

    output_file = os.path.join(directory, f"samples_plot_{method}.png")
    plt.savefig(output_file, dpi=400)
    print(f"{method} samples plot saved to {output_file}")

    plt.close()

def plot_zigzag(directory, method, normalised=False, figsize=(10, 8)):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=figsize)

    event_states_path = os.path.join(directory, f"event_states_{method}.npy")
    event_states = np.load(event_states_path)
    # Create color gradient for the line
    num_points = 200
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_points))
    # Scale data to remove gaps
    x = event_states[0, :num_points]
    y = event_states[1, :num_points]
    # Normalize the data to remove gaps
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
    ax.set_xlabel('Component 0', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component 1', fontsize=12, fontweight='bold')
    plt.title('Zig Zag Sampler Trajectory', fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    plt.tight_layout()
    
    output_file = os.path.join(directory, f"zigzag_plot_{method}.png")
    plt.savefig(output_file, dpi=400)
    print(f"{method} samples plot saved to {output_file}")
    
    plt.close()

def compare_cdf(directory, method1, method2):
     # Identify file paths
     samples1_path = os.path.join(directory, f"samples_{method1}.npy")
     samples2_path = os.path.join(directory, f"samples_{method2}.npy")
     # Load files
     samples1 = np.load(samples1_path).flatten()
     samples2 = np.load(samples2_path).flatten()
     # Create figure and plot
     fig, ax = plt.subplots()
     # Create cdf 1
     sorted_samples1 = np.sort(samples1)
     cdf_values1 = np.arange(1, len(sorted_samples1) + 1) / len(sorted_samples1)
     ax.plot(sorted_samples1, cdf_values1, linestyle = '-', color = 'b', linewidth = 2, label = method1)
     # Create cdf 2
     sorted_samples2 = np.sort(samples2)
     cdf_values2 = np.arange(1, len(sorted_samples2) + 1) / len(sorted_samples2)
     ax.plot(sorted_samples2, cdf_values2, linestyle = '--', color = 'r', label = method2)
     ax.set_xlabel('x')
     ax.set_ylabel('CDF')
     ax.legend()

     output_file = os.path.join(directory, f"cdf_compare_plot_{method1}_{method2}.png")
     plt.savefig(output_file, dpi=400)
     print(f"Compare CDF plot saved to {output_file}")

     plt.close()
     

def pdmp_ks_test(samples, dist, method):
        from scipy.stats import kstest
        stat, p = kstest(samples, dist, args=(np.mean(samples), np.std(samples)))
        if p > 0.05:
            print(f"{method} samples are likely normally distributed, p = {p}")
        else:
            print(f"{method} Samples are likely not normally distributed, p = {p}")

def autocorr(directory, max_lag, method, do_write_autocorr_samples=False, do_plot_autocorr=False):
    max_lag = int(max_lag)
    # Identify file paths
    samples_path = os.path.join(directory, f"samples_{method}.npy")
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
    write_npy(directory, autocorr=f"autocorr_samples_{method}")
    # Plot autocorrelation samples
    if do_plot_autocorr:
        fig, ax = plt.subplots()
        ax.plot(autocorr_samples)
        output_file = os.path.join(directory, f"autocorr_plot_{method}.png")
        plt.savefig(output_file, dpi=400)
        print(f"{method} autocorrelation plot saved to {output_file}")
    # Read existing parameters JSON
    json_path = os.path.join(directory, f"params_{method}.json")
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

    

def compare_autocorr(directory, max_lag, method1, method2, do_write_autocorr_samples=False, do_compare_autocorr=False):
    print(f"Comparing CDFs for {method1} and {method2}...")
    autocorr1, iat1, N_eff1 = autocorr(directory, max_lag, method1, do_write_autocorr_samples, do_compare_autocorr)
    autocorr2, iat2, N_eff2 = autocorr(directory, max_lag, method2, do_write_autocorr_samples, do_compare_autocorr) 
    fig, ax = plt.subplots()
    ax.plot(autocorr1, linestyle = '-', color = 'b', label = f'{method1}:\nIAT = {iat1:.2f}, N_eff = {N_eff1:.0f}')
    ax.plot(autocorr2, linestyle = '--', color = 'r', label = f'{method2}:\nIAT = {iat2:.2f}, N_eff = {N_eff2:.0f}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.legend()
    output_file = os.path.join(directory, f"compare_autocorr_plot_{method1}_{method2}.png")
    plt.savefig(output_file, dpi=400)
    print(f"Compare autocorrelation plot saved to {output_file}")

    plt.close()

def mean_square_displacement(directory, method, mean=0):
    samples_path = os.path.join(directory, f"samples_{method}.npy")
    # Load files
    samples = np.load(samples_path)
    msd = np.mean((samples-mean)**2)
    json_path = os.path.join(directory, f"params_{method}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    # Write IAT to JSON
    data['msd'] = msd 
    with open(json_path, 'w') as f:
         json.dump(data, f, indent=4)
    return msd



"""
Save info in output json e.g. ks test stat, p_value, variance etc
"""
