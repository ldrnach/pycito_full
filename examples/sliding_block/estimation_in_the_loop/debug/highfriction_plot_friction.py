"""
Luke Drnach
May 2, 2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
#TODO: Tune tracking gains, improve kernel prediction performance (sigmoid between RBF and linear kernel?)
from pycito.systems.block.block import Block
from pycito.controller.mpc import LinearizedContactTrajectory

SOURCE = os.path.join('examples','sliding_block','estimation_in_the_loop','high_friction')
FILES = ['mpc', 'campc']
EXT = '_reference.pkl'
STYLES = ['-','--']
block = Block()
block.Finalize()

fig, axs = plt.subplots(1,1)

for file, style in zip(FILES, STYLES):
    lintraj = LinearizedContactTrajectory.loadLinearizedTrajectory(block, os.path.join(SOURCE, file + EXT))
    t = lintraj._time
    mu = [fc[0][0,4] for fc in lintraj.friccone_cstr]
    mu = np.array(mu)
    axs.plot(t, mu, style, linewidth=1.5, label=file)
axs.set_xlabel("Time (s)")
axs.set_ylabel("FrictionCoefficient")
axs.legend()
axs.set_ylim([-0.5, 1.])

plt.show()