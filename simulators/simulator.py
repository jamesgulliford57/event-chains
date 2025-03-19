import numpy as np 
from abc import ABC, abstractmethod
from utils.sim_utils import timer, print_section
from utils.data_utils import write_json, write_npy

class Simulator(ABC):
    """
    Base class for simulators.
    """
    def __init__(self, target, num_samples, x0=0.0, **simulator_specific_params):
        """
        Initialise the simulator.
        
        Parameters:
        ---
        target : Target
            Target distribution to simulate.
        num_samples : int
            Number of samples to generate.
        x0 : float, list
            Initial state.
        simulator_specific_params : dict
            Parameters specific to the chosen simulator.
        """
        self.simulator_name = self.__class__.__name__
        self.target = target
        self.target_name = self.target.__class__.__name__
        self.target_params = self.target.target_params
        self.num_samples = num_samples

        if not isinstance(x0[0], (float, int)):
            raise ValueError(f"Initial state elements must be a float or int. Provided: {x0}")
        elif isinstance(x0, (float, int)):
            x0 = [x0]

        self.x0 = x0
        self.x = np.array(x0)
        for key, value in simulator_specific_params['simulator_specific_params']['simulator_specific_params'].items():
            setattr(self, key, value)

    @timer
    def sim(self, directory):
        """
        Performs simulation and writes simulation output files.
        
        Parameters
        ---
        output_dir: str
            Directory to save output files.
        """
        print_section(f'Running {self.__class__.__name__} simulation with ' 
            f'{self.target.__class__.__name__} target with {self.num_samples} '
            f' and x0 ={self.x0}...')
        # Clear output json
        open(f'{directory}/output.json', 'w').close()
        # Generate samples using selected simulator
        samples = self.sim_chain()

        # Write the samples to a numpy file
        write_npy(directory, **samples)
        # Write output parameters json
        params = {key : value for key, value in self.__dict__.items() if isinstance(value, (int, float, list, str, dict))}
        write_json(directory, **{f"output" : params})

    @abstractmethod
    def sim_chain(self):
        """
        Abstract method for chain simulation methods.
        """
        raise NotImplementedError('Simulator class requires sim_chain method to be implemented.')


