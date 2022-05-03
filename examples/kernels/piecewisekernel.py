import numpy as np
import matplotlib.pyplot as plt
import pycito.systems.kernels as kernels

def piecewise_func(x):
    if x < 0:
        return 0
    elif x < 2:
        return -x**2/4
    else:
        return -1

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

# Reference model
x = np.linspace(-3, 5, 400)
y = np.array([piecewise_func(x_) for x_ in x])

# Sample points
x_samples = np.array([-2, -1, 0, 0.5, 1, 1.5, 2, 3, 4])
y_samples = np.array([piecewise_func(xs) for xs in x_samples])
# Generate the plot
plt.plot(x, y, linewidth=2.0, label='Piecewise Function')
plt.plot(x_samples, y_samples, 'x', label='Samples')
# Build the kernel models
rbf = kernels.RegularizedRBFKernel(length_scale = 0.1, noise = 0.1)
ph = kernels.RegularizedPseudoHuberKernel(length_scale = 1, delta = 0.1, noise = 0.1)
lin = kernels.RegularizedLinearKernel(weights = np.ones((1,)), offset = 1, noise = 0.1)
poly2 = kernels.RegularizedPolynomialKernel(weights = np.ones((1,)), offset = 1, degree = 2, noise = 0.1)
poly4 = kernels.RegularizedPolynomialKernel(weights = np.ones((1,)), offset = 1, degree = 4, noise = 0.1)
hk = kernels.RegularizedHyperbolicKernel(weights = np.ones((1,)), offset = 1, noise = 0.1)
models = [ph, lin, poly2, poly4]
names = ["PseudoHuber", "Linear", "Poly-2", "Poly-4"]
# Regress the data and plot
x_samples = np.atleast_2d(x_samples)
y_samples = np.atleast_2d(y_samples).T
x = np.atleast_2d(x)
for model, name in zip(models, names):
    kmodel = KernelModel(model)
    kmodel.regress(x_samples, y_samples)
    y_pred = kmodel.predict(x)
    plt.plot(np.squeeze(x), np.squeeze(y_pred), linewidth=0.75, label=name)

plt.legend(frameon = False)
plt.show()