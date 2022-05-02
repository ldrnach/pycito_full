import os
import matplotlib.pyplot as plt
from pycito.utilities import load


SOURCE = os.path.join('examples','sliding_block','estimation_in_the_loop','stepterrain')
FILENAME = 'contactambiguity.pkl'

model = load(os.path.join(SOURCE, FILENAME))
fig, axs = plt.subplots(2,1)
# First get the x values of the contact points
x = model.surface._sample_points[0,:]
# Now get the expected surface errors, and upper and lower bounds
axs[0].plot(x, model.surface.model_errors, 'kx', label='Rectified Model')
axs[0].plot(x, model.upper_bound.surface.model_errors, 'r-', label='UpperBound')
axs[0].plot(x, model.lower_bound.surface.model_errors, 'b:', label='LowerBound')
axs[0].set_ylabel('Distance Errors (m)')
axs[0].legend(frameon=False)
# Also show the friction coefficient errors
axs[1].plot(x, model.friction.model_errors, 'kx', label='Rectified Model')
axs[1].plot(x, model.upper_bound.friction.model_errors, 'r-',label='UpperBound')
axs[1].plot(x, model.lower_bound.friction.model_errors, 'b:', label='LowerBound')
axs[1].set_ylabel('Friction Coefficient Errors')
axs[1].set_xlabel('Position (m)')

plt.show()
