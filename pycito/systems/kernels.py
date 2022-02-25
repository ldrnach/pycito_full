"""
kernels: basic implementation of kernel functions 

Luke Drnach
February 22, 2022
"""
from difflib import Differ
import numpy as np
import abc

class DifferentiableStationaryKernel(abc.ABC):
    
    def __call__(self, X, Y = None):
        """
        Evaluate the kernel matrix
        """
        if Y is None:
            return self.eval(X, X)
        else:
            return self.eval(X, Y)

    @staticmethod
    def _reshape_inputs(X, Y):
        return np.reshape(X, (X.shape[0], -1)), np.reshape(Y, (Y.shape[0], -1))

    @staticmethod
    def _squared_distance(X, Y):
        """
        Calculate the squared distance between the two arrays
        
        Arguments:
            x: (n_features, n_samples_x) array of exapmle vectors
            y: (n_features, n_samples_y) array of example vectors

        Return:
            D: (n_samples_x, n_samples_y) array of squared distance values
        """
        assert X.shape[0] == Y.shape[0], f'example vectors x and y must have the same first dimension'
        X, Y = DifferentiableStationaryKernel._reshape_inputs(X, Y)
        return np.sum((X.T[:, None] - Y.T)**2, axis=-1)

    def eval(self, x, y):
        """
        Return the kernel matrix calculated from two example datapoints
    
        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors
        
        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        D = self._squared_distance(x, y)
        return self._eval_stationary(D)

    @abc.abstractmethod
    def _eval_stationary(self, sq_dist):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, x, y):
        raise NotImplementedError

class RBFKernel(DifferentiableStationaryKernel):
    def __init__(self, length_scale = 1.0):
        assert length_scale > 0, "length_scale must be positive"
        self.length_scale = 1.0
        self._scale = -1/(2*self.length_scale **2)

    def _eval_stationary(self, distances):
        """
        Evaluate the kernel, using a matrix of pointwise distances
        
        Arguments:
            distances: (N, M) array of pointwise distances
        
        Return Values:
            (N, M) array of kernel values
        """
        return np.exp(self._scale * distances)

    def gradient(self, x, y):
        """
        Evaluate the gradient of the kernel
        
        Arguments:
            x: (n_features, 1) array. This is argument for which the gradient is calculated
            y: (n_features, n_samples_y) array. This is an example point
        
        Return values:
            grad: (n_samples_y, n_features) array of gradients with respect to the input x
        """
        x, y = self._reshape_inputs(x, y)
        K = self.eval(x, y)
        assert K.shape[0] == 1, f"Gradient calculation supports only a single vector as first input"
        return 2 * self._scale * np.diag(K.flatten()).dot((x - y).transpose())

class PseudoHuberKernel(DifferentiableStationaryKernel):
    def __init__(self, length_scale=1.0, delta=1.0):
        assert length_scale > 0, 'length_scale must be positive'
        assert delta > 0, 'delta must be positive'
        self._length_scale =  length_scale
        self._delta = delta

    def _pseudohuber(self, dist): 
        return np.sqrt(1 + dist/self._delta ** 2 )

    def _eval_stationary(self, dist):
        """
        Evaluate the kernel using the distance between the two sample points
        """
        p = self._pseudohuber(dist)
        return np.exp(-self._delta ** 2 / self._length_scale * (p - 1))

    def gradient(self, x, y):
        """
        Evaluate the gradient of the kernel
        
        Arguments:
            x: (n_features, 1) array. This is argument for which the gradient is calculated
            y: (n_features, n_samples_y) array. This is an example point
        
        Return values:
            grad: (n_samples_y, n_features) array of gradients with respect to the input x
        """
        x, y = self._reshape_inputs(x, y)
        d = self._squared_distance(x, y)
        assert d.shape[0] == 1, "gradient calculation supports only a single vector as first input"
        K = self._eval_stationary(d)
        p = self._pseudohuber(d)
        dK = 1/(self._length_scale * p) * K
        return - np.diag(dK.flatten()).dot((x - y).transpose())

if __name__ == '__main__':
    print("Hello from kernels.py!")