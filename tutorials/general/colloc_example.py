import numpy as np
from trajopt.collocation import RadauCollocation as rc
import matplotlib.pyplot as plt


def polynomial(t):
    x =  3*t**3 - t**2 + 2*t + 1
    dx = 9*t**2 - 2*t + 2
    return x, dx


t = np.linspace(0, 1, 1001)
xpoly, dxpoly = polynomial(t)

interp = rc(order=3, domain=[0,1])
interp.values, _ = polynomial(interp.nodes)

xinterp = interp.eval(t)
dxinterp = interp.derivative(t)

assert np.allclose(xpoly, xinterp), "Interpolation incorrect"
assert np.allclose(dxpoly, dxinterp), "Derivative incorrect"

fig, axs = plt.subplots(2,1)
axs[0].plot(t, xpoly, linewidth=1.5, label='Polynomial')
axs[0].plot(t, xinterp, linewidth=1.5, label='Interpolant')
axs[0].plot(interp.nodes, interp.values, 'o', label='Nodes')
axs[0].set_ylabel('Polynomial')
axs[0].legend()
axs[1].plot(t, dxpoly, linewidth=1.5, label='Polynomial')
axs[1].plot(t, dxinterp, linewidth=1.5, label='Interpolant')
axs[1].set_ylabel('Derivative')
axs[1].set_xlabel('Time')
plt.show()


