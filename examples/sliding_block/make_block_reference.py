import numpy as np
import os
from pycito.utilities import load, save

SOURCEDIR = os.path.join('data','slidingblock')
SOURCE = 'block_trajopt.pkl'
TARGET = 'block_reference.pkl'

data = load(os.path.join(SOURCEDIR, SOURCE))
print(f"Final time T = {data['time'][-1]}")
print(f"Final control u[T] = {data['control'][:, -1]}")
print(f"Final state x[:, T] = {data['state'][:, -1]}")
print(f"Final force f[:, -1] = {data['force'][:, -1]}")

dt = np.mean(np.diff(data['time']))
print(f"average sample time dt = {dt}")
xF = data['state'][:, -1]
uF = data['control'][:, -1]
fF = data['force'][:, -1]
fF[1:] = 0.
T = data['time'][-1]
time = []
state = []
control = []
force = []
while T < 2.0:
    T += dt
    state.append(xF)
    control.append(uF)
    force.append(fF)
    time.append(T)
state = np.column_stack(state)
control = np.column_stack(control)
force = np.column_stack(force)
time = np.asarray(time)
data['state'] = np.concatenate([data['state'], state], axis=1)
data['control'] = np.concatenate([data['control'], control], axis=1)
data['force'] = np.concatenate([data['force'], force], axis=1)
data['time'] = np.concatenate([data['time'], time], axis=0)
print("After extension")
print(f"Final time T = {data['time'][-1]}")
print(f"Final control u[T] = {data['control'][:, -1]}")
print(f"Final state x[:, T] = {data['state'][:, -1]}")
print(f"Final force f[:, -1] = {data['force'][:, -1]}")
save(os.path.join(SOURCEDIR, TARGET), data)