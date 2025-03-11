from simulators.simulator import Simulator
import numpy as np 
from utils.data_utils import write_npy, write_json
from utils.sim_utils import timer
from utils.build_utils import parse_value

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
            return np.random.exponential(1 / self.target.event_rate_bound())
        # Otherwise calculate using analytical event time
        else:
            return self.target.event_time_func(self.x, self.v)

    def _thinned_acceptance_prob(self):
        """
        Returns the acceptance probability when using Poisson thinning.
        """
        return self.target.event_rate() / self.target.event_rate_bound()

    @timer
    def sim(self, output_dir):
        """
        Performs simulation.
        
        Parameters
        ---
        output_dir: str
            Directory to save output files.
        """

        print(f'\nInitiating {self.__class__.__name__} simulation of ' 
              f'{self.target.__class__.__name__} target with {self.num_samples} '
              f'samples and final time {self.final_time}...')
        
        time = 0.0
        events = 0
        event_times = [0]
        event_states = [self.x] 
        while time < self.final_time:
            if events % 10000 == 0 and events > 10000:
                print(f"{len(event_times)} events occured")
            event_time, component_to_flip = self.find_next_event_time()
            self.x = self.x + self.v * event_time
            time += float(event_time)
            event_states.append(self.x.copy())
            event_times.append(time)
            # For thinned simulation, only flip the velocity if the event is accepted
            if self.poisson_thinned:
                if np.random.rand() < self.poisson_thinned_acceptance_prob():
                    self.v = -self.v
                    events += 1
            else:
                # For the non-Poisson-thinned PDMP, always flip the velocity after each event
                self.v[component_to_flip] = -self.v[component_to_flip]
                events += 1  

        print("\nPDMP simulation complete.")

        # Convert to arrays, 1 dimension per row
        event_times = np.array(event_times)
        event_states = np.squeeze(np.array(event_states).T)
        sample_times = np.linspace(0, self.final_time, self.num_samples)
        samples = np.zeros((self.target.dim, self.num_samples))
        # Interpolate between events to generate samples
        if self.target.dim == 1:
            samples = np.interp(sample_times, event_times.ravel(), event_states.ravel())
        else:
            for i in range(self.target.dim):
                samples[i, :] = np.interp(sample_times, event_times, event_states[i, :])

        # Write the samples to a numpy file
        write_npy(output_dir, **{f"samples": samples, f"event_states" : event_states})
        # Write output parameters json
        if self.poisson_thinned:
            # If using thinning, calculate and record the acceptance rate
            self.acceptance_rate = events / len(event_times)
        params = {key : value for key, value in self.__dict__.items() if isinstance(value, (int, float, list, str, dict))}
        write_json(output_dir, **{f"params" : params})


