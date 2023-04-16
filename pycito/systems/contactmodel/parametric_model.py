from __future__ import annotations

import abc

import numpy as np

from configuration.parametricmodel import ConstantModelConfig, FlatModelConfig


class DifferentiableModel(abc.ABC):
    def __call__(self, x):
        return self.eval(x)

    @abc.abstractmethod
    def eval(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, x):
        raise NotImplementedError


class ConstantModel(DifferentiableModel):
    def __init__(self, const=1.0):
        self._const = const

    @classmethod
    def build_from_config(cls, config: ConstantModelConfig) -> ConstantModel:
        return cls(const=config.const)

    def eval(self, x):
        """
        Evalute the constant prior
        Note: the strange syntax ensures the prior works with autodiff types

        Arguments:
            x: (n, ) numpy array

        Return Values:
            out: (1, ) numpy array
        """
        return np.atleast_1d(np.vdot(np.zeros_like(x), x) + self._const)

    def gradient(self, x):
        """
        Evalute the gradient of the prior
        Note: the syntax 0*x ensures the prior works with autodiff types

        Arguments:
            x: (n, ) numpy array

        Return Values:
            out: (1, n) numpy array
        """
        return np.reshape(0 * x, (1, x.shape[0]))


class FlatModel(DifferentiableModel):
    def __init__(self, location=1.0, direction=np.array([0, 0, 1])):
        self._location = location
        self._direction = direction

    @classmethod
    def build_from_config(cls, config: FlatModelConfig) -> FlatModel:
        return cls(location=config.location, direction=np.array(config.direction))

    def eval(self, x):
        """
        Evaluates the flat model, written to work with autodiff types

        Arguments:
            x: (3,) numpy array

        Return values:
            out: (1,) numpy array
        """
        return np.atleast_1d(np.vdot(self._direction, x) - self._location)

    def gradient(self, x):
        """
        Evaluate the gradient of the flat model, written to work with autodiff types

        Arguments:
            x: (3,) numpy array

        Return values:
            out: (1,3) numpy array
        """
        return np.reshape(
            self._direction + np.vdot(np.zeros_like(x), x), (1, self._direction.size)
        )
