from matplotlib import pyplot as plt
import numpy as np

z = np.linspace(0, 100, 101)
x = (z/10)**2
fig, ax = plt.subplots(2,1)
ax[0].plot(z, x, '-x')
ax[0].set_yscale('symlog', linthresh = 1e-4)
ax[1].plot(z, x, '-x')
ax[1].set_yscale('symlog', linthreshy=1e-4)
ax[1].grid(True)
plt.draw()
plt.show()
