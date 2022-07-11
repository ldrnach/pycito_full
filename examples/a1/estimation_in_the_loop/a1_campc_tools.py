import numpy as np
import os
from pycito.utilities import load

SOURCE = os.path.join('examples','a1','simulation_tests','fullstep','timestepping_1e-4')
FILE = 'simdata.pkl'

data = load(os.path.join(SOURCE, FILE))
dt = data['time'][-1] - data['time'][0]
dx = data['state'][:,-1] - data['state'][:, 0]

v = dx[0]/dt
d = 3
print(f"A1 traversed {dx[0]:0.2f}m in {dt:0.2f}s")
print(f"A1 moves at {v:0.2f}m/s")
print(f"To go {d}m, A1 needs to run for {d/v:0.2f}s")