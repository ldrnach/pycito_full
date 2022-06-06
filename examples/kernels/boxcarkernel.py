"""
Check that we can effectively model an offset using a centered linear kernel

"""
import numpy as np
import matplotlib.pyplot as plt
import pycito.systems.kernels as kernels

class KernelModel():
    def __init__(self, kernel):
        self.kernel = kernel
        self.data = None
        self.weights = None

    def regress(self, x, y):
        K = self.kernel(x, x)
        self.data = x
        self.weights = np.linalg.lstsq(K, y, rcond=None)[0]

    def predict(self, x_samples):
        K = self.kernel(x_samples, self.data)
        return K.dot(self.weights)


xdata = np.array([[0, 1, 2, 5,  6,  7, 9, 10]])
ydata = np.array([[1, 1, 1,  -1, -1, -1,  2, 2]])

fig, axs = plt.subplots(1,1)
axs.plot(xdata[0,:], ydata[0,:], 'o', label='data')

bx_kernel = kernels.RegularizedBoxcarKernel(length_scale = 1., width=1.1, noise=0.01)
bx_model = KernelModel(bx_kernel)

bx_model.regress(xdata, ydata.T)

xp = np.atleast_2d(np.linspace(-1, 12, 100))
yp = bx_model.predict(xp).T

axs.plot(xp[0,:], yp[0,:], linewidth=1.5, label='Regression')
axs.legend(frameon=False)
plt.show()