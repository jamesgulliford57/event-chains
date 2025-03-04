import numpy as np
import sys 
sys.path.append('..')
from utils import write_npy, write_json, timer

class RandomWalk:
    def __init__(self, num_samples, x0, sigma_prop, dim, pi, **pi_params):
        self.num_samples = num_samples
        self.x0 = x0
        self.samples = np.zeros((dim, num_samples))
        self.samples[:, 0] = x0
        self.sigma_prop = sigma_prop
        self.dim = dim
        self.pi = pi
        self.pi_params = pi_params
        for key, value in pi_params.items(): 
            setattr(self, key, value)  

    def proposal(self, x):
        return np.random.normal(x, self.sigma_prop) # 2.38 = optimal proposals for minimising IAT

    def acc_prob(self, x, y):
        return min(1, self.pi(y) / self.pi(x))

    @timer
    def sim(self, output_dir):
        print(f"\nInititaing RW simulation with N = {self.num_samples}...")
        accepted = 0
        sim_params = {}
        for i in range(1,self.num_samples):
            if i % 10000 == 0:
                print(f'Simulation {i} complete')
            proposal = self.proposal(self.samples[:, i - 1])
            if np.random.uniform() < self.acc_prob(self.samples[:, i - 1], proposal):
                self.samples[:, i] = proposal 
                accepted += 1
            else:
                self.samples[:, i] = self.samples[:, i - 1]
        print("RW simulation complete. Performing analysis...")
        # Write samples to numpy file
        class_name = self.__class__.__name__
        write_npy(output_dir, **{f"samples_{class_name}": self.samples})
        # Record acceptance rate
        acceptance_rate = accepted / self.num_samples
        # Write output parameters json
        params = {'N' : self.num_samples, 'class' : self.__class__.__name__, 'class' : self.__class__.__name__, 'x0' : self.x0, 'sigma_prop' : self.sigma_prop, 'dim' : self.dim, 'acceptance_rate' : acceptance_rate} | self.pi_params
        write_json(output_dir, **{f"params_{class_name}": params})

