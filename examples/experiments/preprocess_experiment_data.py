import os
import numpy as np
import pycito.utilities as utils

START_TIME = 0.
STOP_TIME = 15.0
DOWNSAMPLING = 1

SOURCE = os.path.join('data','a1_experiment','a1_simulation.pkl')
TARGET = os.path.join('data','a1_experiment','a1_simulation_samples.pkl')

data = utils.load(SOURCE)

time = np.squeeze(data['time'])

start = np.argmax(time > START_TIME)
stop = np.argmax(time > STOP_TIME)

print(f"time[{start}] = {time[start]}")
print(f"time[{stop}] = {time[stop]}")
dt = np.diff(time)
print(f"Average sampling rate: dt = {np.mean(dt)} +- {np.std(dt)}")


#Get a subsample of the data
sample_mask = np.arange(start, stop + DOWNSAMPLING, DOWNSAMPLING)

data_slice = {key: value[:, sample_mask] for key, value in data.items()}
data_slice['time'] = np.squeeze(data_slice['time'])
print(f"There are {data_slice['time'].size} samples in the downsampled data")
dt_slice = np.diff(np.squeeze(data_slice['time']))
print(f"New average sampling rate: dt = {np.mean(dt_slice)} +- {np.std(dt_slice)}")
utils.save(TARGET, data_slice)