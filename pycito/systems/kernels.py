"""
kernels: basic implementation of kernel functions 

Luke Drnach
February 22, 2022
"""
from __future__ import annotations
import numpy as np
import abc
from drake_simulation.configuration.kernel import *
from pycito.tests.unittests.kernels_test import LinearKernelTest


class KernelBase(abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Y=None):
        """Evaluate the Kernel matrix"""
        if Y is None:
            return self.eval(X, X)
        else:
            return self.eval(X, Y)

    @staticmethod
    def _reshape_inputs(*args):
        return [np.reshape(X, (X.shape[0], -1)) for X in args]

    @abc.abstractmethod
    def eval(self, x, y):
        """
        Return the kernel matrix calculated from two example datapoints

        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors

        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values

        _eval is an internal method, and assumes the inputs are specified correctly.
        See also: eval
        """

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


class StationaryKernel(KernelBase):
    def __init__(self, weights=1.0):
        super().__init__()
        if isinstance(weights, (int, float)):
            self.weights = np.array([weights])
        elif isinstance(weights, list):
            self.weights = np.array(weights)
        elif isinstance(weights, np.ndarray):
            self.weights = weights
        else:
            raise TypeError(
                f"weights must be an int or float, or a list or array of ints and floats"
            )

    def __call__(self, X, Y=None):
        """
        Evaluate the kernel matrix
        """
        if Y is None:
            return self.eval(X, X)
        else:
            return self.eval(X, Y)

    def _squared_distance(self, X, Y):
        """
        Calculate the squared distance between the two arrays

        Arguments:
            x: (n_features, n_samples_x) array of exapmle vectors
            y: (n_features, n_samples_y) array of example vectors

        Return:
            D: (n_samples_x, n_samples_y) array of squared distance values
        """
        assert (
            X.shape[0] == Y.shape[0]
        ), f"example vectors x and y must have the same first dimension"
        assert (
            self.weights.shape[0] == 1 or self.weights.shape[0] == X.shape[0]
        ), f"Provided {self.weights.shape[0]} weights and got {X.shape[0]} features"
        X, Y = StationaryKernel._reshape_inputs(X, Y)
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


class RBFKernel(StationaryKernel):
    def __init__(self, length_scale=1.0):
        super().__init__(weights=1 / length_scale)
        self._scale = -1 / (2 * length_scale**2)

    @classmethod
    def build_from_config(cls, config: RBFKernelConfig) -> RBFKernel:
        return cls(length_scale=np.array(config.length_scale))

    @classmethod
    def _eval_stationary(self, distances):
        """
        Evaluate the kernel, using a matrix of pointwise distances

        Arguments:
            distances: (N, M) array of pointwise distances

        Return Values:
            (N, M) array of kernel values
        """
        return np.exp(-1.0 / 2 * distances)

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
        assert (
            K.shape[0] == 1
        ), f"Gradient calculation supports only a single vector as first input"
        return 2 * self._scale * np.diag(K.flatten()).dot((x - y).transpose())

    @property
    def length_scale(self):
        finite = self.weights > 0.0
        ls = np.full(self.weights.shape, np.inf)
        ls[finite] = 1 / self.weights[finite]
        return ls


class PseudoHuberKernel(StationaryKernel):
    def __init__(self, length_scale=1.0, delta=1.0):
        super().__init__(weights=1 / length_scale)
        assert delta > 0, "delta must be positive"
        self._delta = delta

    @classmethod
    def build_from_config(cls, config: PseudoHuberKernelConfig) -> PseudoHuberKernel:
        return cls(length_scale=np.array(config.length_scale), delta=config.delta)

    def _pseudohuber(self, dist):
        return np.sqrt(1 + dist / self._delta**2)

    def _eval_stationary(self, dist):
        """
        Evaluate the kernel using the distance between the two sample points
        """
        p = self._pseudohuber(dist)
        return np.exp(-self._delta**2 * (p - 1))

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
        assert (
            d.shape[0] == 1
        ), "gradient calculation supports only a single vector as first input"
        K = self._eval_stationary(d)
        p = self._pseudohuber(d)
        dK = 1 / p * K
        return -np.diag(dK.flatten()).dot(self.weights * (x - y).transpose())

    @property
    def length_scale(self):
        finite = self.weights > 0.0
        ls = np.full(self.weights.shape, np.inf)
        ls[finite] = 1 / self.weights[finite]
        return ls


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

    @classmethod
    def build_from_config(cls, config: LinearKernelConfig) -> LinearKernel:
        return cls(np.array(config.weights), config.offset)

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

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        if isinstance(val, (int, float)):
            self._weights = np.array([[val]])
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                self._weights = np.diag(val)
            else:
                assert (
                    val.ndim == 2 and val.shape[0] == val.shape[1]
                ), "weights must be a square matrix or list of ints or floats"
                self._weights = val
        else:
            raise ValueError(
                "weights must be either a square matrix, a 1D array, or a scalar"
            )


class CenteredLinearKernel(LinearKernel):
    """
    Implements a weighted, centered linear kernel function:
        k(x,y) = (x-c)^T * W * (y - c)
    where:
        W is a matrix of weights of the same dimensions as x and y
        c is a scalar offset parameter, determined as the average value of y
    """

    def __init__(self, weights=np.ones((1,))):
        super().__init__(weights, offset=0)

    @classmethod
    def build_from_config(
        cls, config: CenteredLinearKernelConfig
    ) -> CenteredLinearKernel:
        return cls(np.array(config.weights))

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
        (c,) = self._reshape_inputs(np.mean(y, axis=1))
        K = (x - c).T.dot(self.weights.dot(y - c))
        return K

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
        (c,) = self._reshape_inputs(np.mean(y, axis=1))
        return (y - c).T.dot(self.weights)


class HyperbolicTangentKernel(LinearKernel):
    """
    A hyperbolic tangent kernel function:
        k(x,y) = tanh(y^T * W * x + c)
    where:
        W is a matrix of weights of the same dimensions as x and y
        c is a scalar offset parameter
    """

    def __init__(self, weights, offset=1.0):
        super().__init__(weights, offset)

    @classmethod
    def build_from_config(
        cls, config: HyperbolicTangentKernelConfig
    ) -> HyperbolicTangentKernel:
        return cls(weights=np.array(config.weights), offset=config.offset)

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

    @classmethod
    def build_from_config(cls, config: PolynomialKernelConfig) -> PolynomialKernel:
        return cls(
            weights=np.array(config.weights), offset=config.offset, degree=config.degree
        )

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
        return K**self.degree

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


class ConstantKernel(KernelBase):
    def __init__(self, const=1.0):
        super().__init__()
        self.const = const

    @classmethod
    def build_from_config(cls, config: ConstantKernelConfig) -> ConstantKernel:
        return cls(const=config.const)

    def eval(self, x, y):
        """
        (Internal Method)
        Return the kernel matrix calculated from two example datapoints

        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors

        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        x, y = self._reshape_inputs(x, y)
        return self.const + 0 * x.T.dot(y)

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
        return 0 * y.T * x.T

    @property
    def const(self):
        return self._const

    @const.setter
    def const(self, val):
        assert (
            isinstance(val, (int, float)) and val > 0
        ), f"const must be a nonnegative int or float"
        self._const = val


class WhiteNoiseKernel(KernelBase):
    def __init__(self, noise=1.0):
        super().__init__()
        self.noise = noise

    @classmethod
    def build_from_config(cls, config: WhiteNoiseKernelConfig) -> WhiteNoiseKernel:
        return cls(noise=config.noise)

    def __call__(self, X, Y=None):
        """Evaluate the kernel matrix"""
        if Y is None:
            (X,) = self._reshape_inputs(X)
            return self.noise * np.eye(X.shape[1])
        else:
            X, Y = self._reshape_inputs(X, Y)
            return np.zeros((X.shape[1], Y.shape[1]))

    def eval(self, x, y):
        """
        (Internal Method)
        Return the kernel matrix calculated from two example datapoints

        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors

        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        x, y = self._reshape_inputs(x, y)
        return 0 * x.T.dot(y)

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
        return 0 * y.T * x.T

    @property
    def noise(self):
        """White noise regularization parameter"""
        return self._noise

    @noise.setter
    def noise(self, val):
        """White noise regularization parameter setter"""
        assert (
            isinstance(val, (int, float)) and val >= 0
        ), f"reg must be a nonnegative int or float"
        self._noise = val


class CompositeKernel(KernelBase):
    def __init__(self, *args):
        assert len(args) >= 1, "Must provide at least one kernel to CompositeKernel"
        for kernel in args:
            assert issubclass(
                type(kernel), KernelBase
            ), f"{type(kernel)} is not a defined kernel"
        self.kernels = args

    def __call__(self, X, Y=None):
        """Evaluate the kernel matrix"""
        return sum(kernel(X, Y) for kernel in self.kernels)

    def eval(self, x, y):
        """
        (Internal Method)
        Return the kernel matrix calculated from two example datapoints

        Arguments:
            x: (n_features, n_samples_x) array of example vectors
            y: (n_features, n_samples_y) array of example vectors

        Returns:
            K: (n_samples_x, n_samples_y) array of kernel values
        """
        X, Y = self._reshape_inputs(x, y)
        return sum([kernel.eval(X, Y) for kernel in self.kernels])

    def gradient(self, x, y):
        """
        Evaluate the gradient of the kernel

        Arguments:
            x: (n_features, 1) array. This is argument for which the gradient is calculated
            y: (n_features, n_samples_y) array. This is an example point

        Return values:
            grad: (n_samples_y, n_features) array of gradients with respect to the input x
        """
        X, Y = self._reshape_inputs(x, y)
        return sum([kernel.gradient(X, Y) for kernel in self.kernels])


class RegularizedRBFKernel(CompositeKernel):
    def __init__(self, length_scale=1.0, noise=0.0):
        rbf = RBFKernel(length_scale=length_scale)
        noise = WhiteNoiseKernel(noise=noise)
        super().__init__(rbf, noise)

    @classmethod
    def build_from_config(
        cls, config: RegularizedRBFKernelConfig
    ) -> RegularizedRBFKernel:
        return cls(length_scale=np.array([config.length_scale]), noise=config.noise)


class RegularizedPseudoHuberKernel(CompositeKernel):
    def __init__(self, length_scale=1.0, delta=1.0, noise=0.0):
        ph = PseudoHuberKernel(length_scale=length_scale, delta=delta)
        noise = WhiteNoiseKernel(noise=noise)
        super().__init__(ph, noise)

    @classmethod
    def build_from_config(
        cls, config: RegularizedPseudoHuberKernelConfig
    ) -> RegularizedPseudoHuberKernel:
        return cls(
            length_scale=np.array(config.length_scale),
            delta=config.delta,
            noise=config.noise,
        )


class RegularizedLinearKernel(CompositeKernel):
    def __init__(self, weights, offset, noise):
        lin = LinearKernel(weights, offset)
        reg = WhiteNoiseKernel(noise)
        super().__init__(lin, reg)

    @classmethod
    def build_from_config(
        cls, config: RegularizedLinearKernelConfig
    ) -> RegularizedLinearKernel:
        return cls(
            weights=np.array(config.weights), offset=config.offset, noise=config.noise
        )


class RegularizedHyperbolicKernel(CompositeKernel):
    def __init__(self, weights, offset, noise):
        hk = HyperbolicTangentKernel(weights, offset)
        reg = WhiteNoiseKernel(noise)
        super().__init__(hk, reg)

    @classmethod
    def build_from_config(
        cls, config: RegularizedHyperbolicKernelConfig
    ) -> RegularizedHyperbolicKernel:
        return cls(
            weights=np.array(config.weights), offset=config.offset, noise=config.noise
        )


class RegularizedPolynomialKernel(CompositeKernel):
    def __init__(self, weights, offset, degree, noise):
        pk = PolynomialKernel(weights, offset, degree)
        reg = WhiteNoiseKernel(noise)
        super().__init__(pk, reg)

    @classmethod
    def build_from_config(
        cls, config: RegularizedPolynomialKernelConfig
    ) -> RegularizedPolynomialKernel:
        return cls(
            weights=np.array(config.weights),
            offset=config.offset,
            degree=config.degree,
            noise=config.noise,
        )


class RegularizedConstantKernel(CompositeKernel):
    def __init__(self, const, noise):
        super().__init__(ConstantKernel(const=const), WhiteNoiseKernel(noise=noise))

    @classmethod
    def build_from_config(
        cls, config: RegularizedConstantKernelConfig
    ) -> RegularizedConstantKernel:
        cls(const=config.const, noise=config.noise)


class RegularizedCenteredLinearKernel(CompositeKernel):
    def __init__(self, weights, noise):
        super().__init__(CenteredLinearKernel(weights), WhiteNoiseKernel(noise))

    @classmethod
    def build_from_config(
        cls, config: RegularizedCenteredLinearKernelConfig
    ) -> RegularizedCenteredLinearKernel:
        return cls(weights=np.array(config.weights), noise=config.noise)


if __name__ == "__main__":
    print("Hello from kernels.py!")
