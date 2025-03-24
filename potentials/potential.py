from abc import ABCMeta, abstractmethod

class Potential(metaclass=ABCMeta):
    """
    Potential for Boltzmann distribution.
    """
    def __init__(self, target_params):
        """
        Constructor for potential class.
        """
        for key, value in target_params.items(): 
            setattr(self, key, value)

    @abstractmethod
    def get_potential(self, current_state):
        """
        Returns potential at current state.

        Parameters
        ---
        current_state : float, list 
            Current state.
        """
        raise NotImplementedError

    def get_potential_difference(self, current_state, proposed_state):
        """
        Returns potential difference between two states.

        Parameters
        ---
        current_state : float, list
            Current state.
        proposed_state : float, list
            Proposed state. 
        """
        return self.get_potential(proposed_state) - self.get_potential(current_state)

    @abstractmethod
    def event_time_func(self, *args):
        """
        Event time function for Boltzmann distribution with potential.
        """