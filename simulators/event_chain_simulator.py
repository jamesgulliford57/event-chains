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
        num_samples : int
            Number of samples to simulate.
        """ 
        super().__init__(target=target, num_samples=num_samples, x0=x0, simulator_specific_params=simulator_specific_params)

        if type(self.v0) == float or type(self.v0) == int:
            self.v0 = [self.v0]
        self.v = np.array(self.v0) 

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
            Component of the state to flip.
        """
        return self.target.event_rate(self.x, self.v)[component_to_flip] / self.target.event_rate_bound(self.x, self.v)[component_to_flip]

    def sim_chain(self):
        """
        Performs simulation loop.
        
        """
        
        time = 0.0
        events = 0
        event_times = [0]
        event_states = [self.x] 
        while time < self.final_time:
            event_time, component_to_flip = self.find_next_event_time()
            self.x = self.x + self.v * event_time
            time += event_time
            event_states.append(self.x.copy())
            event_times.append(time)
            # For thinned simulation, only flip the velocity if the event is accepted
            if self.poisson_thinned:
                if np.random.rand() < self._thinned_acceptance_prob(component_to_flip):
                    self.v[component_to_flip] = -self.v[component_to_flip]
                    events += 1
                    if events % 10000 == 0 and events > 10000:
                        print(f"{events} events occured.")
            else:
                # For the non-Poisson-thinned PDMP, always flip the velocity after each event
                self.v[component_to_flip] = -self.v[component_to_flip]
                events += 1  
                if events % 10000 == 0 and events > 10000:
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
            self.thinned_acceptance_rate = events / len(event_times)

        print(f"Event chain simulation complete. {self.num_samples} samples generated")
        
        samples = {'position_samples' : position_samples, 'event_states' : event_states}
        
        return samples


