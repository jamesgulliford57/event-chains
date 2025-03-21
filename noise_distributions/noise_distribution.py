from abc import ABC, abstractmethod

class NoiseDistribution(ABC):
    """
    Parent class for noise distributions.
    """

    def __init__(self):
        """
        Constructor for the NoiseDistribution class.
        """
        self.noise_distribution_name = self.__class__.__name__

    @abstractmethod
    def propose_new_state(self, current_state):
        """
        Returns candidate state.

        Parameters
        ---
        current_state : list or ndarray
            Current state of chain.
        """
        raise NotImplementedError
    
    @abstractmethod
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
        raise NotImplementedError
