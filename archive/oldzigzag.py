import numpy as np
from utils import write_npy, write_json, timer
from models.super_random_walk import RandomWalk
import math 

class RandomWalk1D(RandomWalk):
    def __init__(self, N, x0):
        super().__init__()
        self.N = N
        self.samples = np.zeros(N)
        self.samples[0] = x0
        self.params = {'N' : N, "x0" : x0}

    def _pi(self, x):
        return 1 / (np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)

    def _rw_proposal(self, x):
        return np.random.normal(x, 2.38) # 2.38 = optimal proposals for minimising IAT

    def _rw_acc_prob(self, x, y):
        return min(1, self._pi(y) / self._pi(x))

    @timer
    def rw_sim(self, output_dir):
        print(f"\nInititaing RW simulation with N = {self.N}...")
        accepted = 0
        for i in range(1,self.N):
            if i % 10000 == 0:
                print(f'Simulation {i} complete')
            proposal = self._rw_proposal(self.samples[i - 1])
            if np.random.uniform() < self._rw_acc_prob(self.samples[i - 1], proposal):
                self.samples[i] = proposal 
                accepted += 1
            else:
                self.samples[i] = self.samples[i - 1]
        print("RW simulation complete. Performing analysis...")
        # Write samples to numpy file
        write_npy(output_dir, rw_samples=self.samples)
        # Record acceptance rate
        acceptance_rate = accepted / self.N
        self.params['acceptance_rate'] = acceptance_rate
        # Write output parameters json
        write_json(output_dir, rw_params = self.params)

class PDMP1D:
    def __init__(self, N, final_time, x0, v0=1, thinned=False):
        self.N = N
        self.final_time = final_time
        self.event_times = []
        self.event_states = []
        self.x = x0
        self.v = v0
        self.thinned = thinned
        self.params = {'N' : N, 'final_time' : final_time, 'x0' : x0, 'v0' : v0, 'thinned' : thinned}

    def _event_rate_bound(self):
        return 5 # Probability of event rate being 7 or larger is 10^-12
    
    def _event_rate(self):
        if np.sign(self.x) == np.sign(self.v):
            return np.abs(self.x) 
        else:
            return 0

    def _find_next_event_time(self):
        # If using Poisson thinned event rate call upper bound rate function
        if self.thinned:
            lambda_rate_bound = self._event_rate_bound()
            return np.random.exponential(1 / lambda_rate_bound)
        # Otherwise calculate using analytical event time
        else:
            a = self.v * self.x 
            if a >= 0:
                return -a + math.sqrt(a**2 - 2 * math.log(1 - np.random.rand()))
            else:
                return -a + math.sqrt(-2 * math.log(1 - np.random.rand()))
    
    def _thinned_acceptance_prob(self):
        return self._event_rate() / self._event_rate_bound()
    
    @timer
    def pdmp_sim(self, output_dir):
        print(f"\nInitiating PDMP simulation with N = {self.N}...")
        accepted = 0  # Only used if self.thinned is True
        time = 0.0
        while time < self.final_time:
            if len(self.event_times) % 10000 == 0:
                print(f"Event {len(self.event_times)} occured")
            event_time = self._find_next_event_time()
            self.x += self.v * event_time
            time += event_time
            self.event_states.append(self.x)
            self.event_times.append(time)
            # For thinned simulation, only flip the velocity if the event is accepted
            if self.thinned:
                if np.random.rand() < self._thinned_acceptance_prob():
                    accepted += 1
                    self.v = -self.v
            else:
                # For the standard PDMP, always flip the velocity after each event
                self.v = -self.v
        print("PDMP simulation complete. Performing analysis...")
        sample_times = np.linspace(0, self.final_time, self.N)
        samples = np.interp(sample_times, self.event_times, self.event_states)
        # Write the samples to a numpy file
        write_npy(output_dir, pdmp_samples=samples)
        if self.thinned:
            # If using thinning, calculate and record the acceptance rate
            acceptance_rate = accepted / len(self.event_times)
            self.params['thinning_acceptance_rate'] = acceptance_rate
        write_json(output_dir, pdmp_params=self.params)


"""
- Make superclass that shares simulation method and separates based on keyword method
- Make pdmp superclass so can test different distributions and dimenions (2d zigzag)
- Generalise to allow any 2 models to be compared e.g. PDMP with thinning vs PDMP without
"""
