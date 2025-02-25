import numpy as np 
from models.super_pdmp import PDMP

class ZigZag2d(PDMP):
    def __init__(self, N, final_time, event_time_func, x0=0.0, v0=1, dim=2, poisson_thinned=False, 
                 event_rate=None, event_rate_bound=None):
        super().__init__(N=N, final_time=final_time, event_time_func=event_time_func, 
                         x0=x0, v0=v0, dim=dim, poisson_thinned=poisson_thinned, event_rate=event_rate, event_rate_bound=event_rate_bound)
        
class GaussZigZag2d(ZigZag2d):
    # Generalise to arbitrary mean and variance once all setup
    def __init__(self, N, final_time, x0=0.0, v0=1, dim=2, poisson_thinned=False, 
                 event_rate=None, event_rate_bound=None):
        super().__init__(N=N, final_time=final_time, event_time_func=self.event_time_func, 
                         x0=x0, v0=v0, dim=dim, poisson_thinned=poisson_thinned, event_rate=self.event_rate, event_rate_bound=self.event_rate_bound)

    def event_time_func(self):
        a = self.v * self.x
        event_times = []
        for cpt in a: 
            if cpt >= 0:
                event_times.append(-cpt + np.sqrt(cpt**2 - 2 * np.log(1 - np.random.rand())))
            else:
                event_times.append(-cpt + np.sqrt(-2 * np.log(1 - np.random.rand()))) 
        return min(event_times), np.argmin(event_times)

    def event_rate_bound(self):
        return 5 # Probability of event rate being 7 or larger is 10^-12
    
    def event_rate(self):
        if np.sign(self.x) == np.sign(self.v):
            return np.abs(self.x) 
        else:
            return 0