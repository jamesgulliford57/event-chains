from simulators.simulator import Simulator
import numpy as np 
from utils.build_utils import list_files_excluding
from noise_distributions.gaussian_noise_distribution import GaussianNoiseDistribution

class MetropolisSimulator(Simulator):
    """
    Base class for Metropolis simulators.
    """
    def __init__(self, target, num_samples, x0, samplers, **simulator_specific_params):
        """
        Constructor for the Metropolis simulator.
        
        Parameters:
        ---
        target : Target
            Target distribution to simulate.
        num_samples : int
            Number of samples to simulate.
        x0 : float, list
            Initial state of the random walk.
        simulator_specific_params : dict
            Parameters specific to the Metropolis simulator.
        """
        super().__init__(target=target, num_samples=num_samples, x0=x0, samplers=samplers, simulator_specific_params=simulator_specific_params)
        
        if globals().get(self.noise_distribution) is None:
            raise ValueError(f"Noise distribution {self.noise_distribution} not found. Available: {list_files_excluding('noise_distributions', 'noise_distribution.py')}")
        if not hasattr(self, 'sigma_noise'):
            raise ValueError("No noise standard deviation provided. Please provide 'sigma_noise' in simulator_specific_params."
                             f" For a d-dimensional isotropic Gaussian target, optimal sigma_noise is 2.38 / sqrt(d).")
        if not isinstance(self.sigma_noise, (int, float)) or self.sigma_noise <= 0:
            raise ValueError(f"Noise standard deviation 'sigma_noise' must be positive integer or float. Provided: {self.sigma_noise}."
                             f" For a d-dimensional isotropic Gaussian target, optimal sigma_noise is 2.38 / sqrt(d).")
        self.noise_distribution = globals().get(self.noise_distribution)(sigma_noise=self.sigma_noise)
        self.noise_distribution_name = self.noise_distribution.__class__.__name__

    def _acc_prob(self, current_state, proposed_state):
        """
        Metropolis acceptance probability.
        
        Parameters:
        ---
        current_state : float, list
            Current state.
        proposed_state : float, list
            Proposed state.
        """
        return min(1, self.target.pdf(proposed_state) / self.target.pdf(current_state) * self.noise_distribution.transition_prob(proposed_state, current_state) / self.noise_distribution.transition_prob(current_state, proposed_state))

    def _sim_chain(self):
        """
        Perform Metropolis simulation.
        """
        accepted = 0
        for chain_index in range(1, self.num_samples):
            if chain_index % 10000 == 0:
                print(f'Step {chain_index} occured.')
            proposed_state = self.noise_distribution.propose_new_state(self.x)
            if np.random.uniform() < self._acc_prob(self.x, proposed_state):
                self.x = proposed_state 
                accepted += 1
            # Generate samples
            for sampler in self.samplers.values():
                sampler.generate_sample(current_state=self.x.copy())
        
        # Convert to arrays
        for sampler in self.samplers.values():
                sampler.samples_to_array()

        # Record acceptance rate
        self.acceptance_rate = accepted / self.num_samples

        print(f"Metropolis simulation complete. {self.num_samples} samples generated")

