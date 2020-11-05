"""
gaussianprocess.py: Gaussian Process Regression tools

Luke Drnach
October 21, 2020
"""

import numpy as np
from math import exp, sin, pi
from functools import partial
import matplotlib.pyplot as plt

class GaussianProcess():
    def __init__(self, xdim=1, mean=None, kernel=None, noise=0.):
        """
        Create a new GaussianProcess object

        Arguments:
            xdim: input dimension of the GP (default: 1)
            mean: prior mean function. Takes 1 argument 
            kernel: kernel function for the GP. Takes 2 arguments
            noise: output observation noise level (default: 0.)

        By default, the prior mean is a constant zero function, the kernel is the squared exponential kernel, and the noise level is 0.
        """
        # Initialize arrays for storing observation data
        self.data_x = []
        self.data_y = []
        # Initialize observation covariance and noise models
        self.cov = None
        self.noise = noise
        # Set the prior mean and kernel functions
        if mean is None:
            self.mean = partial(constant_mean, c=0.)
        else:
            self.mean = mean
        if kernel is None:
            self.kernel = partial(sqr_exp_kernel, M=np.eye(xdim), s=1.)
        else:
            self.kernel = kernel

    def add_data(self, x, y):
        """
        Add samples to the GaussianProcess model

        Arguments:
            x: observed inputs to add to the model
            y: observed response data to add 
        """
        # Calculate the updated covariance matrix
        if self.cov is None:
            self.cov = self.kernel(x, x) + self.noise * np.eye(len(x))
            self.chol = np.linalg.cholesky(self.cov).transpose()
        else:
            # "Rank-1" update to the covariance matrix & cholesky factor
            K1 = self.kernel(self.data, x)
            K2 = self.kernel(x, x) + self.noise * np.eye(len(x))
            C1 = np.linalg.solve(self.chol.transpose(), K1)
            C2 = np.linalg.cholesky(K2 - C1.transpose().dot(C1)).transpose()
            self.chol = np.block([[self.chol, C1],[np.zeros(C1.shape), C2]])
        # Append the new observations to the list
        self.data_x.append(x)
        self.data_y.append(y)

    def prior(self, x):
        """ 
        Return the prior mean and covariance of the GP

        Arguments:
            x: list of query points for the prior
        Return values:
            mu: the prior mean
            S: the prior covariance
        """
        mu = 0
        S = self.kernel(x,x)
        return (mu, S)

    def posterior(self, x):
        """
        Return the posterior mean and covariance of the GP
        
        Arguments:
            x: list of query points for the posterior
        Return Values:
            mu: posterior mean of the GP
            S: posterior covariance of the GP
        """
        L1 = self.chol
        K2 = self.kernel(self.data_x, x)
        K3 = self.kernel(x, x)
        R = np.linalg.solve(L1.transpose(), K2)
        # Posterior mean
        mu = R.transpose().dot(np.linalg.solve(L1.transpose(), self.data_y))
        # Posterior variance
        S = K3 - R.transpose().dot(R)
        return (mu,S)


def constant_mean(x, c):
    return c

def sqr_exp_kernel(x1, x2, M, s):

    K = np.zeros(shape=(len(x1), len(x2)))
    for i in range(0, len(x1)):
        for j in range(0, len(x2)):
            dx = x1[i] - x2[j]
            K[i,j] = s*exp(-dx.dot(np.linalg.solve(M, dx))/2)
    
    return K

def plot_gp(ax, x, mu, S):
    """

    """
    s = np.diag(S)
    ax.plot(x, mu, linewidth=1.5)
    ax.fill_between(x, mu-s, mu+s, alpha=0.3)

if __name__ == "__main__":
    # Create a niosy sinusoid example
    sig = 0.2
    t = np.linspace(0,2*pi, 100)
    x = sin(t)
    # Add noise
    r = sig * np.random.default_rng().normal(0, sig, 100)
    y = x + r
    # Choose 20 points at random
    perm = np.random.default_rng().permutation(100)
    idx = perm[0:20]
    # Model the data as a GP
    kernel = partial(sqr_exp_kernel, M=1, s=1)
    gp = GaussianProcess(xdim=1, kernel=kernel, noise=0.1^2)
    # Add the observations to the GP
    gp.add_data(x=t[idx], y=y[idx])
    # Plot the data, the GP prior, and the GP posterior
    fig, axs = plt.subplots(2,2)
    # Data
    axs[0,0].plot(t[idx], y[idx])
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Samples')
    # GP Prior
    mu, S = gp.prior(t)
    plot_gp(axs[0,1], t, mu, S)
    axs[0,1].plot(t[idx], y[idx])
    # GP Posterior
    mu,S = gp.posterior(t)
    plot_gp(axs[1,0], t, mu, S)
    axs[1,1].plot(t[idx], y[idx])
    # Show the plots
    plt.show()