import numpy as np
from scipy.stats import multivariate_normal as mvnorm

class GaussianNoiseDistribution():
    """
    Propose new state of chain using Gaussian proposal distribution.
    """

    def __init__(self, sigma_noise=0.1):
        """
        Constructor of the GaussianNoiseDistribution class.

        Parameters
        ---
        sigma_prop : float
            Standard deviation of the Gaussian distribution.
        """
        self.sigma_noise = sigma_noise
        self.noise_distribution_name = self.__class__.__name__

    def propose_new_state(self, current_state):
        """
        Returns candidate state.

        Parameters
        ---
        current_state : list or ndarray
            Current state of chain.
        """
        return np.random.normal(current_state, self.sigma_noise)
    
    def transition_prob(self, current_state, proposed_state):
        """
        Returns probability of transitioning from current state to proposed state.

        Parameters
        ---
        current_state : list or ndarray
            Current state of chain.
        proposed_state : list or ndarray
            Proposed state of chain.
        """
        dim = len(current_state)
        return mvnorm.pdf(proposed_state, mean=current_state, cov=self.sigma_noise * np.eye(dim))
