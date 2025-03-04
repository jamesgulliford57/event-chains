import numpy as np
from models.random_walk import RandomWalk

class GaussRandomWalk1d(RandomWalk):
    def __init__(self, num_samples, x0, mu_pi, sigma_pi, dim=1):
        super().__init__(num_samples=num_samples, x0=x0, dim=dim, sigma_prop=2.38*sigma_pi, pi=self.pi, mu_pi=mu_pi, sigma_pi=sigma_pi)
        self.mu_pi = mu_pi 
        self.sigma_pi = sigma_pi 

    def pi(self, x):
        return 1 / (np.sqrt(2 * np.pi * self.sigma_pi)) * np.exp(-(x - self.mu_pi) ** 2 / (2 * self.sigma_pi))

class StandardGaussRandomWalk1d(GaussRandomWalk1d):
    def __init__(self, num_samples, x0, mu_pi=0, sigma_pi=1):
        super().__init__(num_samples=num_samples, x0=x0, mu_pi=mu_pi, sigma_pi=sigma_pi)



