from configuration import FlatModelConfig, RBFKernelConfig, SemiparametricModelConfig
from configuration.build_from_config import build_from_config
from pycito.systems.contactmodel import (
    FlatModel,
    SemiparametricModel,
    semiparametric_model,
)
from pycito.systems.kernels import RBFKernel


def test_build_semiparametric_model() -> None:
    prior = FlatModelConfig(location=0.0, direction=(0, 0, 1.0))
    kernel = RBFKernelConfig(length_scale=[1, 1, 1])
    config = SemiparametricModelConfig(prior=prior, kernel=kernel)
    model = build_from_config(semiparametric_model, config)
    assert isinstance(model, SemiparametricModel)
    assert isinstance(model.prior, FlatModel)
    assert isinstance(model.kernel, RBFKernel)
