from simulators.simulator import Simulator
import numpy as np 
from utils.data_utils import write_npy

class EventChainSimulator(Simulator):
    """
    Base class for event chain simulator.
    """
    def __init__(self, target, num_samples, x0, samplers, **simulator_specific_params):
        """
        Constructor for the event chain simulator.
        
        Parameters:
        ---
        target : Target
            Target distribution to simulate.
        num_samples : int
            Number of samples to simulate.
        x0 : float, list
            Initial state of the chain.
        simulator_specific_params : dict
            Parameters specific to the event chain simulator.
        """ 
        super().__init__(target=target, num_samples=num_samples, x0=x0, samplers=samplers, simulator_specific_params=simulator_specific_params)

        if not hasattr(self, 'v0'):
            raise ValueError("No initial velocity provided. Please provide 'v0' in simulator_specific_params.")
        if isinstance(self.v0, (float, int)):
            self.v0 = [self.v0]
        if not all(isinstance(v0_cpt, (float, int)) for v0_cpt in self.v0):
            raise ValueError(f"Initial velocity elements must be a float or int. Provided: {self.v0}")
        if len(self.x0) != len(self.v0):
            raise ValueError(f"State and velocity must be the same dimension. Provided: {self.x0}, {self.v0}")
    
        if not hasattr(self, 'final_time'):
            raise ValueError("Final time not specified. Please provide 'final_time' in simulator_specific_params.")
        if not isinstance(self.final_time, (int, float)) or self.final_time <= 0:
            raise ValueError(f"Final time must be a positive integer or float. Provided: {self.final_time}")

        if not hasattr(self, 'poisson_thinned'):
            print("No Poisson thinning specified. Setting to False.")
            self.poisson_thinned = False
        
        
        self.v = np.array(self.v0) 


    def _find_next_event_time(self):
        """
        Returns the time until the next event and the component of the state to flip.
        """
        # If using Poisson thinned event rate call upper bound rate function
        if self.poisson_thinned:
            event_rate_bounds = self.target.event_rate_bound(self.x, self.v)
            event_times = np.random.exponential(1 / event_rate_bounds)
            component_to_flip = np.argmin(event_times)
            return float(event_times[component_to_flip]), component_to_flip
        # Otherwise calculate using analytical event time
        else:
            event_time, component_to_flip = self.target.event_time_func(self.v, self.x)
            return float(event_time), component_to_flip

    def _thinned_acceptance_prob(self, component_to_flip):
        """
        Returns the acceptance probability when using Poisson thinning.

        Parameters
        ---
        component_to_flip : int
            Component of the state to flip at event.
        """
        return self.target.event_rate(self.x, self.v)[component_to_flip] / self.target.event_rate_bound(self.x, self.v)[component_to_flip]

    def _sim_chain(self):
        """
        Performs ecmc simulation.
        """
        time = 0.0
        events = 0
        event_times = [0.0]
        # Simulate Poisson thinned chain if specified
        if self.poisson_thinned:
            proposed_events = 0
            while time < self.final_time:
                event_time, component_to_flip = self._find_next_event_time()
                self.x = self.x + self.v * event_time
                time += event_time
                proposed_events += 1
                if np.random.rand() < self._thinned_acceptance_prob(component_to_flip):
                    event_times.append(time)
                    self.v[component_to_flip] = -self.v[component_to_flip]
                    events += 1
                    for sampler in self.samplers.values():
                        sampler.generate_sample(current_state=self.x.copy())
                    
                if proposed_events % 10000 == 0 and proposed_events > 0:
                        print(f"{proposed_events} events proposed. {events} events accepted.")
            
        else:
            # Simulate chain
            while time < self.final_time:
                event_time, component_to_flip = self._find_next_event_time()
                self.x = self.x + self.v * event_time
                time += event_time
                event_times.append(time)
                self.v[component_to_flip] = -self.v[component_to_flip]
                events += 1  
                if events % 10000 == 0 and events > 0:
                    print(f"{events} events occured.")
                # Generate samples
                for sampler in self.samplers.values():
                    sampler.generate_sample(current_state=self.x.copy())
                
        # Write event states
        event_times = np.array(event_times)
        # Convert to arrays
        for sampler in self.samplers.values():
                sampler.samples_to_array()
        # Interpolate between events to generate samples
        for sampler in self.samplers.values():
                    sampler.interpolate_samples(final_time=self.final_time, event_times=event_times)
        # If using thinning, calculate and record the acceptance rate
        if self.poisson_thinned:
            self.thinned_acceptance_rate = events / proposed_events

        print(f"Event chain simulation complete. {self.num_samples} samples generated")
        


