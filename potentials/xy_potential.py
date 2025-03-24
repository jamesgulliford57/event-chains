from potentials.potential import Potential
import numpy as np 

class XyPotential(Potential):
    """
    XY potential for Boltzmann distribution.
    """
    def __init__(self, target_params):
        """
        Constructor for XY potential class.
        """
        super().__init__(target_params=target_params)

    def get_neighbours(self, current_state):
        pass

    def get_potential(self, current_state):
        """
        Returns potential at current state.

        Parameters
        ---
        current_state : float, list 
            Current state.
        """
        return 