from simulators.simulator import Simulator
import numpy as np 

class EventChainSimulator(Simulator):
    """
    Base class for event chain simulator.
    """
    def __init__(self, target, num_samples, x0, **simulator_specific_params):
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
        super().__init__(target=target, num_samples=num_samples, x0=x0, simulator_specific_params=simulator_specific_params)

        if not hasattr(self, 'v0'):
            print("No initial velocity provided. Setting to 1.0.")
            self.v0 = 1.0
        if not hasattr(self, 'poisson_thinned'):
            print("No Poisson thinning specified. Setting to False.")
            self.poisson_thinned = False
        if not hasattr(self, 'final_time'):
            raise ValueError("Final time not specified. Please provide 'final_time' in simulator_specific_params.")
        
        if isinstance(self.v0, (float, int)):
            self.v0 = [self.v0]
        self.v = np.array(self.v0) 

        if len(self.x0) != len(self.v0):
            raise ValueError("State and velocity must be the same dimension.")

    def find_next_event_time(self):
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

    def sim_chain(self):
        """
        Performs ecmc simulation.
        
        """
        # Reset state and time variables
        self.x = np.array(self.x0)
        self.v = np.array(self.v0)
        time = 0.0
        events = 0
        event_times = [0.0]
        event_states = [self.x] 
        # Simulate Poisson thinned chain if specified
        if self.poisson_thinned:
            proposed_events = 0
            while time < self.final_time:
                event_time, component_to_flip = self.find_next_event_time()
                self.x = self.x + self.v * event_time
                time += event_time
                proposed_events += 1
                if np.random.rand() < self._thinned_acceptance_prob(component_to_flip):
                    event_states.append(self.x.copy())
                    event_times.append(time)
                    self.v[component_to_flip] = -self.v[component_to_flip]
                    events += 1
                if proposed_events % 10000 == 0 and proposed_events > 0:
                        print(f"{proposed_events} events proposed. {events} events accepted.")
            
        else:
            # Simulate chain
            while time < self.final_time:
                event_time, component_to_flip = self.find_next_event_time()
                self.x = self.x + self.v * event_time
                time += event_time
                event_states.append(self.x.copy())
                event_times.append(time)

                self.v[component_to_flip] = -self.v[component_to_flip]
                events += 1  
                if events % 10000 == 0 and events > 0:
                    print(f"{events} events occured.")

        # Convert to arrays, 1 dimension per row
        event_times = np.array(event_times)
        event_states = np.squeeze(np.array(event_states).T)
        sample_times = np.linspace(0, self.final_time, self.num_samples)
        position_samples = np.zeros((self.target.dim, self.num_samples))
        # Interpolate between events to generate samples
        if self.target.dim == 1:
            position_samples = np.interp(sample_times, event_times.ravel(), event_states.ravel())
        else:
            for i in range(self.target.dim):
                position_samples[i, :] = np.interp(sample_times, event_times, event_states[i, :])
        # If using thinning, calculate and record the acceptance rate
        if self.poisson_thinned:
            self.thinned_acceptance_rate = events / proposed_events

        print(f"Event chain simulation complete. {self.num_samples} samples generated")
        
        samples = {'position_samples' : position_samples, 'event_states' : event_states}
        
        return samples


