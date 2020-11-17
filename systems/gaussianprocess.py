"""
gaussianprocess.py: Gaussian Process Regression tools

Luke Drnach
October 21, 2020
"""

import numpy as np
from math import pi
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
            self.mean = ConstantFunc(0.)
        else:
            self.mean = mean
        if kernel is None:
            self.kernel = SquaredExpKernel(M=np.eye(xdim), s=1.)
        else:
            self.kernel = kernel

    def add_data(self, x, y):
        """
        Add samples to the GaussianProcess model

        Arguments:
            x: observed inputs to add to the model
            y: observed response data to add 
        """
        # Calculate the residuals
        r = np.zeros((y.shape[0], x.shape[1]))
        for n in range(0, x.shape[1]):
            r[:,n] = y[:,n] - self.mean(x[:,n])
        # Calculate the updated covariance matrix
        if self.cov is None:
            self.cov = self.kernel(x, x) + self.noise * np.eye(x.shape[1])
            self.chol = np.linalg.cholesky(self.cov).transpose()
            self.data_x = x
            self.data_y = y
            self.residual = r
        else:
            # "Rank-1" update to the covariance matrix & cholesky factor
            K1 = self.kernel(self.data, x)
            K2 = self.kernel(x, x) + self.noise * np.eye(len(x))
            C1 = np.linalg.solve(self.chol.transpose(), K1)
            C2 = np.linalg.cholesky(K2 - C1.transpose().dot(C1)).transpose()
            self.chol = np.block([[self.chol, C1],[np.zeros(C1.shape), C2]])
            # Append the new observations to the list
            self.data_x = np.concatenate((self.data_x, x), axis=1)
            self.data_y = np.concatenate((self.data_y, y), axis=1)
            self.residual = np.concatenate((self.residual, r), axis=1)

    def pop(self):
        """Remove data from the GP model using last-in, first-out model"""
        if self.data_x.shape[0] == 0:
            return
        elif self.data_x.ndim == 1:
            x = self.data_x
            y = self.data_y
            self.data_x = np.array([])
            self.data_y = np.array([])
            self.residual = np.array([])
            self.chol = np.array([])
            self.cov = None
        else:
            x = self.data_x[:,:-1]
            y = self.data_y[:,:-1]
            # Remove the last datum    
            self.data_x = self.data_x[:,:-1]
            self.data_y = self.data_y[:,:-1]
            self.residual = self.residual[:,:-1]
            # Downdate the cholesky factorization
            self.chol = self.chol[:-1,:-1]
            self.cov = self.cov[:-1,:-1]
        return (x, y)

    def prior(self, x):
        """ 
        Return the prior mean and covariance of the GP

        Arguments:
            x: list of query points for the prior
        Return values:
            mu: the prior mean
            S: the prior covariance
        """
        mu = np.zeros((self.data_y.shape[0], x.shape[1]))
        for n in range(0, x.shape[1]):
            mu[:,n] = self.mean(x[:,n])
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
        mu, K3 = self.prior(x)
        L1 = self.chol
        K2 = self.kernel(self.data_x, x)
        R = np.linalg.solve(L1.transpose(), K2)
        # Posterior mean
        mu = mu + R.transpose().dot(np.linalg.solve(L1.transpose(), self.residual.transpose())).transpose()
        # Posterior variance
        S = K3 - R.transpose().dot(R)
        return (mu,S)

class ConstantFunc():
    def __init__(self, const):
        self.const = const
    
    def __call__(self, x):
        return self.const

class SquaredExpKernel():
    def __init__(self, M, s):
        self.M = M
        self.s = s
    
    def __call__(self, x1, x2):
        K = np.zeros(shape=(x1.shape[1], x2.shape[1]))
        for i in range(0, K.shape[0]):
            for j in range(0, K.shape[1]):
                dx = x1[:,i] - x2[:,j]
                K[i,j] = self.s*np.exp(-dx.dot(np.linalg.solve(self.M, dx))/2)
        return K

def plot_gp(ax, x, mu, S):
    """

    """
    s = np.squeeze(np.diag(S))
    mu = np.squeeze(mu)
    x = np.squeeze(x)
    ax.plot(x, mu, linewidth=1.5)
    ax.fill_between(x, mu-s, mu+s, alpha=0.3)

if __name__ == "__main__":
    # Create a niosy sinusoid example
    sig = 0.2
    t = np.linspace(0,2*pi, 100)
    x = np.sin(t)
    # Add noise
    r = sig * np.random.default_rng().normal(0, sig, 100)
    y = x + r
    # Make sure the arrays have the correct shape
    x = np.expand_dims(x, axis=1).transpose()
    y = np.expand_dims(y, axis=1).transpose()
    t = np.expand_dims(t, axis=1).transpose()
    # Choose 20 points at random
    perm = np.random.default_rng().permutation(100)
    idx = perm[0:20]
    # Model the data as a GP
    kernel = SquaredExpKernel(M=np.ones((1,1)), s=1.)
    gp = GaussianProcess(xdim=1, kernel=kernel, noise=0.1**.2)
    # Add the observations to the GP
    gp.add_data(x=t[:,idx], y=y[:,idx])
    # Plot the data, the GP prior, and the GP posterior
    fig, axs = plt.subplots(2,2)
    # Data
    axs[0,0].plot(t[:,idx], y[:,idx],'bx')
    axs[0,0].plot(np.squeeze(t),np.squeeze(x),'r-')
    axs[0,0].set_xlabel('Sample')
    axs[0,0].set_ylabel('Response')
    # GP Prior
    mu, S = gp.prior(t)
    plot_gp(axs[0,1], t, mu, S)
    axs[0,1].plot(t[:,idx], y[:,idx],'bx')
    axs[0,1].set_xlabel('Sample')
    axs[0,1].set_ylabel('Response')
    # GP Posterior
    mu,S = gp.posterior(t)
    plot_gp(axs[1,0], t, mu, S)
    axs[1,0].plot(t[:,idx], y[:,idx],'bx')
    axs[1,0].set_xlabel('Sample')
    axs[1,0].set_ylabel('Response')
    # GP posterior with the true function
    plot_gp(axs[1,1], t, mu, S)
    axs[1,1].plot(np.squeeze(t), np.squeeze(x), 'r-', linewidth=1.5)
    axs[1,1].set_xlabel('Sample')
    axs[1,1].set_ylabel('Response')
    # Show the plots
    plt.show()