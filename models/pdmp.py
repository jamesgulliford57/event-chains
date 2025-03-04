import numpy as np
import sys 
sys.path.append('..')
from utils import write_npy, write_json, timer

class PDMP:
    def __init__(self, num_samples, final_time, event_time_func, x0=0.0, v0=1, dim=1, poisson_thinned=False, event_rate=None, event_rate_bound=None, **pi_params):
        """
        Simulates a Piecewise Deterministic Markov Process (PDMP).
        
        Parameters
        ---
        num_samples: int 
            Number of samples to simulate.
        """
        if type(x0) == float or type(x0) == int:
            x0 = [x0]
        if type(v0) == float or type(v0) == int:
            v0 = [v0]
        if len(x0) != dim:
            raise ValueError(f'Dimension of initial state and model dimension' 
                             f'are not equal: dim(inital_state) = {len(x0)}, dim(model) = {dim}')
        
        self.num_samples = num_samples
        self.final_time = final_time
        self.event_time_func = event_time_func
        self.x0 = x0 # To save to output parameter file
        self.x = np.array(x0)
        self.v = np.array(v0)
        self.dim = dim 
        self.poisson_thinned = poisson_thinned
        if poisson_thinned:
            self.event_rate = event_rate
            self.event_rate_bound = event_rate_bound
        self.pi_params = pi_params
        for key, value in pi_params.items(): 
            setattr(self, key, value) 

    def _find_next_event_time(self):
        """
        Returns the time until the next event and the component of the state to flip.
        """
        # If using Poisson thinned event rate call upper bound rate function
        if self.poisson_thinned:
            lambda_rate_bound = self.event_rate_bound()
            return np.random.exponential(1 / lambda_rate_bound)
        # Otherwise calculate using analytical event time
        else:
            return self.event_time_func()
    
    def thinned_acceptance_prob(self):
        """
        Returns the acceptance probability when using Poisson thinning.
        """
        return self.event_rate() / self.event_rate_bound()
    
    @timer
    def sim(self, output_dir):
        """
        Performs the PDMP simulation.
        
        Parameters
        ---
        output_dir: str
            Directory to save output files.
        """

        print(f"\nInitiating PDMP simulation with final_time = {self.final_time}...")
        
        time = 0.0
        events = 0
        event_times = []
        event_states = [] 
        while time < self.final_time:
            if events % 10000 == 0 and events > 10000:
                print(f"{len(event_times)} events occured")
            event_time, component_to_flip = self._find_next_event_time()
            self.x = self.x + self.v * event_time
            time += event_time
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

        print("PDMP simulation complete. Performing analysis...")

        # Convert to arrays, 1 dimension per row
        event_times = np.array(event_times)
        event_states = np.squeeze(np.array(event_states).T)
        sample_times = np.linspace(0, self.final_time, self.num_samples)
        samples = np.zeros((self.dim, self.num_samples))
        # Interpolate between events to generate samples
        print(sample_times.shape, event_times[:, 0].shape, event_states.shape)
        if self.dim == 1:
            samples = np.interp(sample_times, event_times.ravel(), event_states.ravel())
        else:
            for i in range(self.dim):
                samples[i, :] = np.interp(sample_times, event_times, event_states[i, :])

        # Write the samples to a numpy file
        class_name = self.__class__.__name__
        write_npy(output_dir, **{f"samples_{class_name}": samples, f"event_states_{class_name}" : event_states})
        # Write output parameters json
        params = {'N' : self.num_samples, 'final_time' : self.final_time, 'class' : self.__class__.__name__, 'x0' : self.x0, 'poisson_thinned' : self.poisson_thinned} | self.pi_params
        if self.poisson_thinned:
            # If using thinning, calculate and record the acceptance rate
            acceptance_rate = events / len(event_times)
            params['thinning_acceptance_rate'] = acceptance_rate
        write_json(output_dir, **{f"params_{class_name}": params})
        #write_npy(output_dir, pdmp_samples=samples) (Delete when worked out don't need)
