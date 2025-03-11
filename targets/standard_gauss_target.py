import math 
import numpy as np 
from targets.target import Target
from scipy.stats import multivariate_normal as mvnorm

class StandardGaussTarget(Target):
    """
    Standard Gaussian target in 1D.
    """
    def __init__(self, dim, target_params={}):
        """
        Initialise standard Gaussian target.
        """
        for key, value in target_params.items():
            setattr(self, key, value)
            
        super().__init__(event_time_func=self.event_time_func,  
                         event_rate=self.event_rate, 
                         event_rate_bound=self.event_rate_bound, 
                         dim=dim, target_params=target_params)

    def pdf(self, x):
        return mvnorm.pdf(x, mean=np.zeros(self.dim), cov=np.eye(self.dim))
    
    def event_time_func(self, x, v):
        a = x * v
        event_times = []
        for cpt in a: 
            if cpt >= 0:
                event_times.append(-cpt + np.sqrt(cpt**2 - 2 * np.log(1 - np.random.rand())))
            else:
                event_times.append(-cpt + np.sqrt(-2 * np.log(1 - np.random.rand()))) 
        return min(event_times), np.argmin(event_times)
        
    def event_rate_bound(self):
        return 5 # Probability of event rate being 7 or larger is 10^-12
    
    def event_rate(self, x, v):
        if np.sign(x) == np.sign(v):
            return np.abs(x)
        else:
            return 0