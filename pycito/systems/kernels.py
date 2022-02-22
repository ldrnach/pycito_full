"""
kernels: basic implementation of kernel functions 

Luke Drnach
February 22, 2022
"""
import numpy as np
import abc

#TODO Unittest

class DifferentiableKernelBase(abc.ABC):
    
    def __call__(self, X, Y = None):
        if Y is None:
            return self._symmetric_kernel_matrix(X)
        else:
            return self._kernel_matrix(X, Y)

    def _kernel_matrix(self, X, Y):
        """
        Arguments:
            X: (n_samples_x, n_features) set of feature vectors
            Y: (n_samples_y, n_features) set of feature vectors
        
        Return values:
            K: (n_samples_x, n_samples_y) kernel matrix
        """
        assert X.shape[1] == Y.shape[1], 'X and Y must have the same number of columns'
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = self.eval(x, y)
        return K

    def _symmetric_kernel_matrix(self, X):
        """
        Arguments:
            X: (n_samples_x, n_features) set of feature vectors
        Return values:
            K: (n_samples_x, n_samples_x) symmetric positive-semidefinite kernel matrix
        """
        K = np.zeros((X.shape[0], X.shape[0]))
        for i, x in enumerate(X):
            K[i, i] = 0.5 * self.eval(x, x)
            for j in range(i+1, X.shape[1]):
                K[i, j] = self.eval(x, X[j, :])
        # Symmetrize
        return K + K.transpose()

    @abc.abstractmethod
    def eval(self, x, y):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, x, y):
        raise NotImplementedError

class RBFKernel(DifferentiableKernelBase):
    def __init__(self, length_scale = 1.0):
        assert length_scale > 0, "length_scale must be positive"
        self.length_scale = 1.0
        self._scale = -1/(2*self.length_scale **2)

    def eval(self, x, y):
        """Evaluate the kernel"""
        return np.exp(self._scale * (x - y).dot(x - y))

    def gradient(self, x, y):
        """Evaluate the gradient of the kernel"""
        K = self.eval(x, y)
        return 2 * self._scale * K * (x - y).transpose()

class PseudoHuberKernel(DifferentiableKernelBase):
    def __init__(self, length_scale=1.0, delta=1.0):
        assert length_scale > 0, 'length_scale must be positive'
        assert delta > 0, 'delta must be positive'
        self._length_scale =  length_scale
        self._delta = delta

    def _pseudohuber(self, x, y): 
        return np.sqrt(1 + (x - y).dot(x-y)/self._delta ** 2 )

    def eval(self, x, y):
        """Evaluate the kernel"""
        p = self._pseudohuber(x, y)
        return np.exp( - self._delta ** 2 / self._length_scale * (p - 1))

    def gradient(self, x, y):
        """Evaluate the gradient of the kernel"""
        K = self.eval(x, y)
        p = self._pseudohuber(x, y)
        return - 1 /(self._length_scale * p) * K * (x - y).transpose()

if __name__ == '__main__':
    print("Hello from kernels.py!")