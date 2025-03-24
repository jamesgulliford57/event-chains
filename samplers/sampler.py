from abc import ABCMeta, abstractmethod
import numpy as np

class Sampler(metaclass=ABCMeta):
    """
    Class for taking observations of system during simulation.
    """
    def __init__(self, num_samples, dim, x0):
        """
        Constructor for sampler class.

        Parameters
        ---
        num_samples : int,
            Number of samples to generate.
        dim : int,
            Dimension of target.
        x0 : float, list
            Initial state.
        """
        self.num_samples = num_samples
        self.dim = dim
        self.samples = None 
        self.x0 = x0
        self.array_name = None
    
    def initialise_samples(self):
        """
        Generate arrays to store samples.
        """
        self.samples = []
        self.generate_sample(self.x0)
    
    @abstractmethod
    def generate_sample(self, current_state):
        """
        Generate position sample from current state of chain.

        Parameters
        ---
        current_state : float or np.ndarray
            Current state of chain.
        """
        raise NotImplementedError
    
    def samples_to_array(self):
        """
        Convert samples list into array post-simulation.

        Parameters
        ---
        samples_list : list
            List containing samples generated from simulation.
        """
        self.samples = np.atleast_2d(np.squeeze(np.array(self.samples).T))
        

    def interpolate_samples(self, final_time, event_times):
        """
        Interpolate event state to obtain final samples for ECMC simulations.

        final_time : float
            Time at which ECMC simulation commenced.
        event_times : np.ndarray
            Times at which ECMC events occured.
        """
        sample_times = np.linspace(0, final_time, self.num_samples)
        dim_samples = np.shape(self.samples)[0]

        interp_samples = np.zeros((dim_samples, self.num_samples))

        if dim_samples == 1:
            interp_samples = np.interp(sample_times, event_times.ravel(), self.samples.ravel())
        else:
            for i in range(dim_samples):
                interp_samples[i, :] = np.interp(sample_times, event_times, self.samples[i, :])
        
        self.samples = interp_samples