from scipy.stats import multivariate_normal as mvnorm
import numpy as np 
from targets.target import Target

class GaussTarget(Target):
    """
    Gaussian target.
    """
    def __init__(self, dim, target_params={}):
        """
        Constructor for the Gaussian target.

        Parameters
        ---
        dim : int
            Dimension of the target distribution.
        target_params : dict
            Parameters specific to the target distribution.
        """
        super().__init__(event_time_func=self.event_time_func,  
                         event_rate=self.event_rate, 
                         event_rate_bound=self.event_rate_bound, 
                         dim=dim, target_params=target_params)
    
    def pdf(self, x):
        """
        Probability density function of the target distribution.
        
        Parameters
        ---
        x : float, list
            Input to the pdf.
        """
        return mvnorm.pdf(x, mean=self.mu_target*np.ones(self.dim), cov=self.sigma_target**2*np.eye(self.dim))
    
    def event_time_func(self, x, v):
        """
        Event time function for the Gaussian target.

        Parameters
        ---
        x : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        a = (x - self.mu_target) * v
        event_times = []
        for cpt in a: 
            if cpt >= 0:
                event_times.append(-cpt + np.sqrt(cpt**2 - 2 * self.sigma_target**2 * np.log(1 - np.random.rand())))
            else:
                event_times.append(-cpt + np.sqrt(-2 * self.sigma_target**2 * np.log(1 - np.random.rand()))) 
        return min(event_times), np.argmin(event_times)
    
    def event_rate_bound(self, x, v):
        """
        Event rate bound for the Gaussian target for Poisson thinning.

        Parameters
        ---
        x : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        return 5 * np.ones(self.dim)
    
    def event_rate(self, x, v):
        """
        Event rate for the Gaussian target.

        Parameters
        ---
        x : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        event_rates = []
        for x_cpt, v_cpt in zip(x, v):
            event_rate = np.maximum(0.0, v_cpt * (x_cpt - self.mu_target) / self.sigma_target**2)
            event_rates.append(event_rate)
        return np.array(event_rates)
        
    