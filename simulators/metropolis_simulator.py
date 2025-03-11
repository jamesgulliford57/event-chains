from simulators.simulator import Simulator
import numpy as np 
from utils.data_utils import write_npy, write_json
from utils.sim_utils import timer
from utils.build_utils import parse_value
from noise_dists.gaussian_noise_dist import GaussianNoiseDistribution

class MetropolisSimulator(Simulator):
    """
    Base class for Metropolis simulators.
    """
    def __init__(self, target, num_samples, x0, **simulator_specific_params):
        """
        Constructor for the Metropolis simulator.
        
        Parameters:
        ---
        num_samples : int
            Number of samples to simulate.
        x0 : float, list
            Initial state of the random walk.
        sigma_prop : float
            Standard deviation of the proposal distribution.
        dim : int
            Dimension of the random walk.
        simulator_specific_params : dict
            Parameters specific to the Metropolis simulator.
        """
        super().__init__(target=target, num_samples=num_samples, x0=x0, simulator_specific_params=simulator_specific_params)
        
        self.noise_distribution = globals().get(self.noise_distribution)(sigma_noise=self.sigma_noise)
        self.noise_distribution_name = self.noise_distribution.__class__.__name__

    def _acc_prob(self, x, y):
        """
        Metropolis acceptance probability.
        
        Parameters:
        ---
        x : float, list
            Current state of the random walk.
        y : float, list
            Proposed state of the random walk.
        """
        return min(1, self.target.pdf(y) / self.target.pdf(x) * self.noise_distribution.transition_prob(y, x) / self.noise_distribution.transition_prob(x, y))

    @timer
    def sim(self, output_dir):
        """
        Perform random walk simulation.
    
        Parameters:
        ---
        output_dir : str
            Directory to save output files.
        """

        print(f'\nInitiating {self.__class__.__name__} simulation of ' 
              f'{self.target.__class__.__name__} target with {self.num_samples} '
              f'samples')

        accepted = 0
        samples = [self.x]
        for i in range(1, self.num_samples):
            if i % 10000 == 0:
                print(f'Simulation {i} complete')
            proposed_state = self.noise_distribution.propose_new_state(self.x)
            if np.random.uniform() < self._acc_prob(self.x, proposed_state):
                self.x = proposed_state 
                accepted += 1
            samples.append(self.x.copy())

        print("\nMetropolis simulation complete. Performing analysis...")
        
        samples = np.array(samples).T 
        # Write samples to numpy file
        write_npy(output_dir, **{f"samples": samples})
        # Record acceptance rate
        self.acceptance_rate = accepted / self.num_samples
        # Write output parameters json
        params = {key : value for key, value in self.__dict__.items() if isinstance(value, (int, float, list, str, dict))}
        #params = {'N' : self.num_samples, 'class' : self.__class__.__name__, 'class' : self.__class__.__name__, 'x0' : self.x0, 'sigma_prop' : self.sigma_prop, 'dim' : self.dim, 'acceptance_rate' : acceptance_rate} | self.pi_params
        write_json(output_dir, **{f"params": params})

