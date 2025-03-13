from abc import ABC, abstractmethod

class Target(ABC):
    """
    Target (stationary) distribution for simulation.
    """
    def __init__(self, event_time_func, event_rate=None, event_rate_bound=None, dim=1, target_params={}):
        """
        Constructor for the target distribution.
        
        Parameters
        ---
        event_time_func : function
            Function to calculate the time until the next event.
        event_rate : function
            Function to calculate the event rate.
        event_rate_bound : function
            Function to calculate the upper bound of the event rate.
        dim : int
            Dimension of the target distribution.
        target_params : dict
            Parameters specific to the target distribution.
        """
        
        self.event_time_func = event_time_func 
        self.event_rate = event_rate
        self.event_rate_bound = event_rate_bound
        self.target_params = target_params
        for key, value in target_params.items(): 
            setattr(self, key, value) 

        @abstractmethod 
        def pdf(*args):
            """
            Outputs probability density at provided input.
            """
            pass 


    

