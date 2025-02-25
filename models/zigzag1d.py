import numpy as np
from models.super_pdmp import PDMP
import math 

class ZigZag1d(PDMP):
    def __init__(self, N, final_time, event_time_func, x0=0.0, v0=1, dim=1, poisson_thinned=False, 
                 event_rate=None, event_rate_bound=None, **pi_params):
        super().__init__(N=N, final_time=final_time, event_time_func=event_time_func, 
                         x0=x0, v0=v0, dim=dim, poisson_thinned=poisson_thinned, event_rate=event_rate, event_rate_bound=event_rate_bound, **pi_params)

class GaussZigZag1d(ZigZag1d):
    # Generalise to arbitrary mean and variance once all setup
    def __init__(self, N, final_time, x0=0.0, v0=1, dim=1, poisson_thinned=False, 
                 event_rate=None, event_rate_bound=None, mu_pi=0, sigma_pi=1):
        super().__init__(N=N, final_time=final_time, event_time_func=self.event_time_func, 
                         x0=x0, v0=v0, dim=dim, poisson_thinned=poisson_thinned, event_rate=self.event_rate, 
                         event_rate_bound=self.event_rate_bound, mu_pi=mu_pi, sigma_pi=sigma_pi)

    def event_time_func(self):
        a = self.v * self.x 
        if a >= 0:
            return -a + math.sqrt(a**2 - 2 * math.log(1 - np.random.rand()))
        else:
            return -a + math.sqrt(-2 * math.log(1 - np.random.rand())) 
        
    def event_rate_bound(self):
        return 5 # Probability of event rate being 7 or larger is 10^-12
    
    def event_rate(self):
        if np.sign(self.x) == np.sign(self.v):
            return np.abs(self.x) 
        else:
            return 0

#class StandardGaussZigZag1d(GaussZigZag1D):    
