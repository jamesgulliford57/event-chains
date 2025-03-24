from abc import ABCMeta, abstractmethod
import numpy as np
from potentials.potential import Potential

class HarmonicPotential(Potential, metaclass=ABCMeta):
    """
    Harmonic potential for Boltzmann distribution.
    """
    def __init__(self, target_params):
        """
        Constructor for harmonic potential class.
        """
        super().__init__(target_params=target_params)

    def get_potential(self, current_state):
        """
        Returns potential at current state.

        Parameters
        ---
        current_state : float, list 
            Current state.
        """
        return 0.5 * self.mass * self.omega**2 * np.dot(current_state, current_state)

    def event_time_func(self, current_state, v):
        """
        Event time function for the Boltzmann target with harmonic potential.

        Parameters
        ---
        current_state : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        a = v * self.beta * self.mass * self.omega**2 * current_state
        event_times = []
        for cpt in a: 
            if cpt >= 0:
                event_times.append((-cpt + np.sqrt(cpt**2 - 2 * self.beta * self.mass * self.omega**2 * np.log(1 - np.random.rand()))) / (self.beta * self.mass * self.omega**2))
            else:
                event_times.append((-cpt + np.sqrt(-2 * self.beta * self.mass * self.omega**2 * np.log(1 - np.random.rand()))) / (self.beta * self.mass * self.omega**2)) 
        return min(event_times), np.argmin(event_times) 
    
    def event_rate_bound(self, current_state, v):
        """
        Event rate bound for the Gaussian target for Poisson thinning.

        Parameters
        ---
        x : float, list
            Current state.
        v : float, list
            Current velocity.
        """
        return 5 * self.beta * self.mass * self.omega**2 * np.ones(self.dim)

    def event_rate(self, current_state, v):
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
        for x_cpt, v_cpt in zip(current_state, v):
            event_rate = np.maximum(0.0, v_cpt * x_cpt * self.beta * self.mass * self.omega**2)
            event_rates.append(event_rate)
        return np.array(event_rates)
    