"""
Tools for modeling contact semiparametrically

Luke Drnach
February 16, 2022
"""
#TODO: Unittesting
#TODO: Offically subclass SemiparametricContactModel

import numpy as np
import abc
import pycito.systems.kernels as kernels

class DifferentiablePrior(abc.ABC):
    def __call__(self, x):
        return self.eval(x)

    @abc.abstractmethod
    def eval(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, x):
        raise NotImplementedError

class ConstantPrior(DifferentiablePrior):
    def __init__(self, const = 1.0):
        self._const = 1.0

    def eval(self, x):    
        """
        Evalute the constant prior
        Note: the strange syntax ensures the prior works with autodiff types
        """
        return np.zeros_like(x).dot(x) + self._const

    def gradient(self, x):
        """
        Evalute the gradient of the prior
        Note: the syntax 0*x ensures the prior works with autodiff types
        """
        return 0 * x

class FlatSurfacePrior(DifferentiablePrior):
    def __init__(self, height = 1.0, direction = np.array([0, 0, 1])):
        self._height = 1.0
        self._direction = direction

    def eval(self, x):
        """
        Evaluates the flat terrain prior, written to work with autodiff types
        """
        return self._direction.dot(x) - self._height

    def gradient(self, x):
        """
        Evaluate the gradient of the flat terrain prior, written to work with autodiff types
        """
        return self._direction + 0*x

class SemiparametricModel():
    def __init__(self, prior, kernel):
        assert issubclass(type(prior), DifferentiablePrior), "prior must be a concrete implementation of DifferentiablePrior"
        assert issubclass(type(kernel), kernels.DifferentiableKernelBase), "kernel must be a concrete implementation of DifferentiableKernelBase"
        self.prior = prior
        self.kernel = kernel
        self._kernel_weights = None
        self._sample_points = None

    def add_samples(self, samples, weights):
        """
        adds samples, and the relevant weights, to the model

        Arguments:
            samples (n_samples, n_features): numpy array of sample points
            weights (n_samples, ): numpy array of corresponding kernel weights
        """
        assert samples.shape[0] == weights.shape[0], 'there must be as many weights as there are samples'
        if self._sample_points is not None:
            assert samples.shape[1] == self._sample_points.shape[1], 'new samples must have the same number of features as the existing samples'
            self._sample_points = np.concatenate([self._sample_points, samples], axis=0)
            self._weights = np.concatenate([self._weights, weights], axis=0)
        else:
            self._sample_points = samples
            self._weights = weights        

    def eval(self, x):
        """
        Evaluate the semiparametric model at the sample point
        
        Arguments:
            x: (n_features,) numpy array, the sample point at which to evaluate the model
        """
        # Evaluate the prior
        y = self.prior(x)
        # Evaluate the kernel
        if self._sample_points is not None:
            y += self.kernel(self._sample_points, x).dot(self._kernel_weights)
        return y

    def gradient(self, x):
        """
        Evaluate the gradient of the semiparametric model

        Arguments:
            x: (n_features,) numpy array, the sample point at which to evaluate the model
        """
        # Evaluate the gradient of the prior
        dy = self.prior.gradient(x)
        # Evaluate the kernel gradient
        if self._sample_points is not None:
            for point, weight in zip(self._sample_points, self._kernel_weights):
                dy += self.kernel.gradient(x, point) * weight
        return dy

    @property
    def model_errors(self):
        if self._sample_points is None:
            return None
        K = self.kernel(self._sample_points)
        return K.dot(self._kernel_weights)

class SemiparametricContactModel():    
    def __init__(self, surface_model, friction_model):
        assert isinstance(surface_model, SemiparametricModel), 'surface_model must be a semiparametric model'
        assert isinstance(friction_model, SemiparametricModel), 'friction_model must be a semiparametric model'
        self.surface = surface_model
        self.friction = friction_model

    @classmethod
    def FlatSurfaceWithRBFKernel(cls, height=0., friction=1., lengthscale=0.1):
        """
        Factory method for constructing a semiparametric contact model
        
        Assumes the prior is a flat surface with constant friction
        uses independent RBF kernels for the surface and friction models
        """
        surf = SemiparametricModel(prior = FlatSurfacePrior(height = height),
                                    kernel = kernels.RBFKernel(length_scale = lengthscale))
        fric = SemiparametricModel(prior = ConstantPrior(const = friction),
                                    kernel = kernels.RBFKernel(length_scale = lengthscale))
        return cls(surf, fric)

    @classmethod
    def FlatSurfaceWithHuberKernel(cls, height = 0., friction = 1., lengthscale = 0.1, delta = 0.1):
        """
        Factory method for constructing a semiparametric contact model
        Assumes the prior is a flat surface with constant friction
        Uses independent Pseudo-Huber kernels for the surface and friction models
        """
        surf = SemiparametricModel(prior = FlatSurfacePrior(height = height),
                                    kernel = kernels.PseudoHuberKernel(lengthscale, delta))
        fric = SemiparametricModel(prior = ConstantPrior(friction),
                                    kernel = kernels.PseudoHuberKernel(lengthscale, delta))
        return cls(surf, fric)

    def add_samples(self, sample_points, surface_weights, friction_weights):
        self.surface.add_samples(sample_points, surface_weights)
        self.friction.add_samples(sample_points, friction_weights)


    def eval_surface(self, x):
        return self.surface(x)

    def eval_friction(self, x):
        return self.friction(x)

    

if __name__ == '__main__':
    print("Hello from semi-parametric contact!")
