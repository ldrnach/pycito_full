from __future__ import annotations

import numpy as np

from drake_simulation.configuration.build_from_config import build_from_config
from drake_simulation.configuration.parametricmodel import PiecewiseModelConfig

from . import parametric_model


class PiecewiseModel(parametric_model.DifferentiableModel):
    def __init__(self, breaks, models):
        # Input checking
        assert isinstance(breaks, list), f"breaks must be a list of numeric values"
        assert isinstance(
            models, list
        ), f"models must be a list of DifferentiableModel types"
        assert len(breaks) + 1 == len(
            models
        ), f"there must be one more model than the number of break points"
        breaks.append(np.inf)
        breaks = np.array(breaks)
        assert np.all(
            breaks[1:] >= breaks[:-1]
        ), f"breaks must be monotonically increasing"
        # Setup
        self._breaks = breaks
        self._models = models

    @classmethod
    def build_from_config(cls, config: PiecewiseModelConfig) -> PiecewiseModel:
        models = [build_from_config(parametric_model, model) for model in config.models]
        return cls(breaks=config.breaks, models=models)

    def eval(self, x):
        """
        Evaluate the piecewise model

        Arguments:
            x: (3, ) numpy array

        Return values:
            out: (1, ) numpy array
        """
        return self.get_submodel(x).eval(x)

    def gradient(self, x):
        """
        Evaluate the gradient of the piecewise model, written to work with autodiff types

        Arguments:
            x: (3,) numpy array

        Return values:
            out: (1, 3) numpy array
        """
        return self.get_submodel(x).gradient(x)

    def get_submodel(self, x):
        """
        Return the part of the piecewise model that corresponds to the current point
        """
        return self._models[np.argmax(x[0] < self._breaks)]
