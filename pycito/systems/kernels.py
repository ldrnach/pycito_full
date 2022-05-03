"""
kernels: basic implementation of kernel functions 

Luke Drnach
February 22, 2022
"""
import numpy as np
import abc

class KernelBase(abc.ABC):
    def __call__(self, X, Y=None):
        """Evaluate the Kernel matrix"""
        if Y is None:
            return self.eval(X, X)
        else:
            return self.eval(X, Y)

    @staticmethod
    def _reshape_inputs(X, Y):
        return np.reshape(X, (X.shape[0], -1)), np.reshape(Y, (Y.shape[0], -1))

    @abc.abstractmethod
    def eval(self, x, y):
        """
        Return the kernel matrix calculated from two example datapoints
    
        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors
        
        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, x, y):
        """
        Evaluate the gradient of the kernel
        
        Arguments:
            x: (n_features, 1) array. This is argument for which the gradient is calculated
            y: (n_features, n_samples_y) array. This is an example point
        
        Return values:
            grad: (n_samples_y, n_features) array of gradients with respect to the input x
        """
        raise NotImplementedError


class DifferentiableStationaryKernel(abc.ABC):
    def __init__(self, weights=1., reg=0.):
        super().__init__()
        assert isinstance(reg, (int, float)) and reg >= 0., 'reg must be a nonnegative integer or float'
        if isinstance(weights, (int, float)):
            self.weights = np.array([weights])
        elif isinstance(weights, list):
            self.weights = np.array(weights)
        elif isinstance(weights, np.ndarray):
            self.weights = weights
        else:
            raise TypeError(f"weights must be an int or float, or a list or array of ints and floats")
        
        self.reg = reg

    def __call__(self, X, Y = None):
        """
        Evaluate the kernel matrix
        """
        if Y is None:
            K = self.eval(X, X)
            return K + self.reg * np.eye(K.shape[0])
        else:
            return self.eval(X, Y)

    @staticmethod
    def _reshape_inputs(X, Y):
        return np.reshape(X, (X.shape[0], -1)), np.reshape(Y, (Y.shape[0], -1))

    def _squared_distance(self, X, Y):
        """
        Calculate the squared distance between the two arrays
        
        Arguments:
            x: (n_features, n_samples_x) array of exapmle vectors
            y: (n_features, n_samples_y) array of example vectors

        Return:
            D: (n_samples_x, n_samples_y) array of squared distance values
        """
        assert X.shape[0] == Y.shape[0], f'example vectors x and y must have the same first dimension'
        assert self.weights.shape[0] == 1 or self.weights.shape[0] == X.shape[0], f"Provided {self.weights.shape[0]} weights and got {X.shape[0]} features"
        X, Y = DifferentiableStationaryKernel._reshape_inputs(X, Y)
        D3 = X.T[:, None] - Y.T
        D3 = D3 * self.weights[None, None, :]
        return np.sum(D3**2, axis=-1)

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
    def __init__(self, length_scale = 1.0, reg=0.):
        super().__init__(weights = 1/length_scale, reg = reg)
        self._scale = -1/(2*length_scale **2)

    def _eval_stationary(self, distances):
        """
        Evaluate the kernel, using a matrix of pointwise distances
        
        Arguments:
            distances: (N, M) array of pointwise distances
        
        Return Values:
            (N, M) array of kernel values
        """
        return np.exp(-1./2* distances)

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

    @property
    def length_scale(self):
        finite = self.weights > 0.
        ls = np.full(self.weights.shape, np.inf)
        ls[finite] = 1/self.weights[finite]
        return ls

class PseudoHuberKernel(DifferentiableStationaryKernel):
    def __init__(self, length_scale=1.0, delta=1.0, reg=0.):
        super().__init__(reg=reg)
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

class LinearKernel(KernelBase):
    """
    Implements a weighted linear kernel function:
        k(x,y) = y^T * W * x + c
    where:
        W is a matrix of weights of the same dimensions as x and y 
        c is a scalar offset parameter    
    """

    def __init__(self, weights=np.ones((1,)), offset=1):
        super().__init__()
        self.weights = weights
        self.offset = offset

    def eval(self, x, y):
        """
        Return the kernel matrix calculated from two example datapoints
    
        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors
        
        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        x, y = self._reshape_inputs(x, y)
        K = x.T.dot(self.weights.dot(y))
        return K + self.offset

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
        return y.T.dot(self.weights)

class HyperbolicTangentKernel(LinearKernel):
    """
    A hyperbolic tangent kernel function:
        k(x,y) = tanh(y^T * W * x + c)
    where:
        W is a matrix of weights of the same dimensions as x and y 
        c is a scalar offset parameter    
    """
    def __init__(self, weights, offset=1.):
        super().__init__(weights, offset)
        
    def eval(self, x, y):
        """
        Return the kernel matrix calculated from two example datapoints
    
        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors
        
        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        K = super().eval(x, y)
        return np.tanh(K)

    def gradient(self, x, y):
        """
        Evaluate the gradient of the kernel
        
        Arguments:
            x: (n_features, 1) array. This is argument for which the gradient is calculated
            y: (n_features, n_samples_y) array. This is an example point
        
        Return values:
            grad: (n_samples_y, n_features) array of gradients with respect to the input x
        """
        dK = super().gradient(x, y)
        K = self.eval(x, y)
        return (1 - K**2).T * dK

class PolynomialKernel(LinearKernel):
    """
    A hyperbolic tangent kernel function:
        k(x,y) = tanh(y^T * W * x + c)
    where:
        W is a matrix of weights of the same dimensions as x and y 
        c is a scalar offset parameter    
    """
    def __init__(self, weights, offset=1, degree=2):
        super().__init__(weights, offset)
        self.degree = degree

    def eval(self, x, y):
        """
        Return the kernel matrix calculated from two example datapoints
    
        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors
        
        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        K = super().eval(x, y)
        return K ** self.degree

    def gradient(self, x, y):
        """
        Evaluate the gradient of the kernel
        
        Arguments:
            x: (n_features, 1) array. This is argument for which the gradient is calculated
            y: (n_features, n_samples_y) array. This is an example point
        
        Return values:
            grad: (n_samples_y, n_features) array of gradients with respect to the input x
        """
        dK = super().gradient(x, y)
        K = super().eval(x, y)
        return self.degree * K.T ** (self.degree - 1) * dK

if __name__ == '__main__':
    print("Hello from kernels.py!")