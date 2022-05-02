"""
Luke Drnach
April 25, 2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from pycito.systems.block.block import Block
from pycito.controller.mpc import LinearizedContactTrajectory

SOURCE = os.path.join('examples','sliding_block','estimation_in_the_loop','stepterrain')
FILES = ['mpc', 'campc']
EXT = '_reference.pkl'
STYLES = ['-','--']
block = Block()
block.Finalize()

fig, axs = plt.subplots(1,1)

for file, style in zip(FILES, STYLES):
    lintraj = LinearizedContactTrajectory.loadLinearizedTrajectory(block, os.path.join(SOURCE, file + EXT))
    t = lintraj._time
    d = [dist[1] for dist in lintraj.distance_cstr]
    d = np.array(d)
    axs.plot(t, d, style, linewidth=1.5, label=file)
axs.set_xlabel("Time (s)")
axs.set_ylabel("Normal Distance (m)")
axs.legend()
axs.set_ylim([-0.5, 1.])

plt.show()
