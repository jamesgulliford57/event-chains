from samplers.sampler import Sampler

class SquaredDisplacementSampler(Sampler):
    """
    Class for taking observations of position during simulation.
    """
    def __init__(self, num_samples, dim, x0):
        """
        Constructor for position sampler class.
        """
        super().__init__(num_samples=num_samples, dim=dim, x0=x0)
        self.array_name = 'squared_displacement_samples'
    
    def generate_sample(self, current_state):
        """
        Generate position sample from current state of chain.

        Parameters
        ---
        current_state : float or np.ndarray
            Current state of chain.
        """
        self.samples.append(sum(cpt**2 for cpt in current_state))