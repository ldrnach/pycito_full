from __future__ import annotations

import numpy as np
from parametric_model import ConstantModel, DifferentiableModel, FlatModel

import pycito.systems.kernels as kernels
from drake_simulation.configuration.semiparametricmodel import SemiparametricModelConfig

from . import parametric_model


class SemiparametricModel(DifferentiableModel):
    def __init__(self, prior, kernel):
        assert issubclass(
            type(prior), DifferentiableModel
        ), "prior must be a concrete implementation of DifferentiableModel"
        assert issubclass(
            type(kernel), kernels.KernelBase
        ), "kernel must be a concrete implementation of KernelBase"
        self.prior = prior
        self.kernel = kernel
        self._kernel_weights = None
        self._sample_points = None

    @classmethod
    def build_from_config(
        cls, config: SemiparametricModelConfig
    ) -> SemiparametricModel:
        prior = getattr(parametric_model, config.prior.type).build_from_config(
            config.prior
        )
        kernel = getattr(kernels, config.kernel.type).build_from_config(config.kernel)
        return cls(prior, kernel)

    @classmethod
    def ConstantPriorWithRBFKernel(cls, const=0, length_scale=1, reg=0.0):
        return cls(
            prior=ConstantModel(const=const),
            kernel=kernels.RegularizedRBFKernel(length_scale=length_scale, noise=reg),
        )

    @classmethod
    def FlatPriorWithRBFKernel(
        cls, location=0.0, direction=np.array([0, 0, 1]), length_scale=1.0, reg=0.0
    ):
        return cls(
            prior=FlatModel(location=location, direction=direction),
            kernel=kernels.RegularizedRBFKernel(length_scale=length_scale, noise=reg),
        )

    @classmethod
    def ConstantPriorWithHuberKernel(cls, const=0, length_scale=1, delta=1, reg=0.0):
        return cls(
            prior=ConstantModel(const=const),
            kernel=kernels.RegularizedPseudoHuberKernel(
                length_scale=length_scale, delta=delta, noise=reg
            ),
        )

    @classmethod
    def FlatPriorWithHuberKernel(
        cls,
        location=0,
        direction=np.array([0, 0, 1]),
        length_scale=1.0,
        delta=1,
        reg=0.0,
    ):
        return cls(
            prior=FlatModel(location, direction),
            kernel=kernels.RegularizedPseudoHuberKernel(length_scale, delta, noise=reg),
        )

    def add_samples(self, samples, weights):
        """
        adds samples, and the relevant weights, to the model

        Arguments:
            samples (n_features, n_samples): numpy array of sample points
            weights (n_samples, ): numpy array of corresponding kernel weights
        """
        samples = np.atleast_2d(samples)
        assert (
            samples.shape[1] == weights.shape[0]
        ), "there must be as many weights as there are samples"
        if self._sample_points is not None:
            assert (
                samples.shape[0] == self._sample_points.shape[0]
            ), "new samples must have the same number of features as the existing samples"
            self._sample_points = np.concatenate([self._sample_points, samples], axis=1)
            self._kernel_weights = np.concatenate(
                [self._kernel_weights, weights], axis=0
            )
        else:
            self._sample_points = samples
            self._kernel_weights = weights

    def eval(self, x):
        """
        Evaluate the semiparametric model at the sample point

        Arguments:
            x: (n_features,) numpy array, the sample point at which to evaluate the model

        Return Values:
            y: (1,) the output of the model
        """
        # Evaluate the prior
        y = self.prior(x)
        # Evaluate the kernel
        if self._sample_points is not None:
            y = y + self.kernel(x, self._sample_points).dot(self._kernel_weights)
        return y

    def gradient(self, x):
        """
        Evaluate the gradient of the semiparametric model

        Arguments:
            x: (n_features,) numpy array, the sample point at which to evaluate the model

        Return value:
            grad (1, n_features) numpy array, the gradient of the model at the sample point
        """
        # Evaluate the gradient of the prior
        dy = self.prior.gradient(x)
        dy = np.reshape(dy, (dy.shape[0], -1))
        # Evaluate the kernel gradient
        if self._sample_points is not None:
            dk = self.kernel.gradient(x, self._sample_points)
            w = np.reshape(self._kernel_weights, (self._kernel_weights.shape[0], -1))
            dy = dy + w.T.dot(dk)
        return dy

    def compress(self, atol=1e-4):
        """
        Eliminates sample points with little effect on the overall model.

        Compress deletes all sample points and kernel weights if the corresponding kernel weight is less than the tolerance (default: 1e-4)
        """
        if self._sample_points is None:
            return
        idx = np.abs(self._kernel_weights) > atol
        self._sample_points = np.copy(self._sample_points[:, idx])
        self._kernel_weights = np.copy(self._kernel_weights[idx])

    @property
    def model_errors(self):
        if self._sample_points is None:
            return None
        K = self.kernel(self._sample_points)
        return K.dot(self._kernel_weights)

    @property
    def num_samples(self):
        if self._sample_points is None:
            return 0
        else:
            return self._sample_points.shape[1]
