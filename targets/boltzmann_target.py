from targets.target import Target 
import numpy as np 
from abc import ABCMeta, abstractmethod

class BoltzmannTarget(Target, metaclass=ABCMeta):
    """
    Boltzmann distribution target.
    """
    def __init__(self, dim, target_params={}):
        """
        Constructor for the Boltzmann target.

        Parameters
        ---
        dim : int
            Dimension of the target distribution.
        target_params : dict
            Parameters specific to the Botzmann distribution.
        """
        super().__init__(event_time_func=self.event_time_func,  
                         event_rate=self.event_rate, 
                         event_rate_bound=self.event_rate_bound, 
                         dim=dim, target_params=target_params)
        
    def pdf(self, current_state):
        """
        Probability density function of the Botlzmann distribution.
        
        Parameters
        ---
        current_state : float, list
            Current state
        """
        return np.exp(-self.beta * self.potential.get_potential(current_state=current_state))

    def pdf_ratio(self, current_state, proposed_state):
        """
        Ratio of probability density functions of the Botlzmann distribution for two statess.
        
        Parameters
        ---
        current_state : float, list
            Current state.
        proposed_state : float, list
            Proposed state.
        """
        return np.exp(-self.beta * self.get_potential_difference(current_state=current_state, proposed_state=proposed_state))
    
    def event_time_func(self, current_state, v):
        """
        Event time function for the Gaussian target.

        Parameters
        ---
        current_state : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        return self.potential.event_time_func(current_state=current_state, v=v)
    
    def event_rate_bound(self, current_state, v):
        """
        Event rate bound for the Gaussian target for Poisson thinning.

        Parameters
        ---
        current_state : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        if hasattr(self.potential, 'event_rate_bound'):
            event_rate_bound = self.potential.event_rate_bound(current_state=current_state, v=v)
        else:
            raise NotImplementedError(f"{self.potential.__class__.__name__} must implement `event_rate_bound` if Poisson thinning is to be used.")

        return event_rate_bound
    
    def event_rate(self, current_state, v):
        """
        Event rate for the Gaussian target.

        Parameters
        ---
        current_state : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        if hasattr(self.potential, 'event_rate'):
            event_rate = self.potential.event_rate(current_state=current_state, v=v)
        else:
            raise NotImplementedError(f"{self.potential.__class__.__name__} must implement `event_rate` if Poisson thinning is to be used.")
        return event_rate

        
