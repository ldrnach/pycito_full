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

x = np.atleast_2d(np.linspace(3,5,10))
print(x.shape)
y = -1 * np.ones_like(x) + np.random.default_rng().normal(0, 0.01, 10)
print(y.shape)
fig, axs = plt.subplots(1,1)
axs.plot(x[0,:], y[0,:], 'o', label='data')

axs.set_xlim([0,7])
axs.set_ylim([-2,2])

cl_kernel = kernels.RegularizedCenteredLinearKernel(weights = np.ones((1,)), noise = 0.01)
cst_kernel = kernels.ConstantKernel(const = 1)
kernel = kernels.CompositeKernel(cl_kernel, cst_kernel)
model = KernelModel(kernel)
model.regress(x, y.T)
x_p = np.atleast_2d(np.linspace(0, 7, 100))
y_p = model.predict(x_p).T

axs.plot(x_p[0,:], y_p[0,:], linewidth=1.5, label='Regression')


axs.legend(frameon= False)
plt.show()