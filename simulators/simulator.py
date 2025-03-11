import numpy as np 

class Simulator():
    """
    Base class for simulators.
    """
    def __init__(self, target, num_samples, x0=0.0, **simulator_specific_params):
        """
        Initialise the simulator.
        
        Parameters:
        ---
        num_samples : int
            Number of samples to simulate.
        """
        self.simulator_name = self.__class__.__name__
        self.target = target
        self.target_name = self.target.__class__.__name__
        self.target_params = self.target.target_params
        self.num_samples = num_samples

        if type(x0) == float or type(x0) == int:
            x0 = [x0]
        if len(x0) != target.dim:
            raise ValueError(f'Dimension of initial state and model dimension' 
                             f'are not equal: dim(inital_state) = {len(x0)}, dim(model) = {target.dim}')

        self.x0 = x0
        self.x = np.array(x0)
        for key, value in simulator_specific_params['simulator_specific_params']['simulator_specific_params'].items():
            setattr(self, key, value)