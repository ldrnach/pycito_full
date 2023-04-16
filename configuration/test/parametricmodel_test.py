import pycito.systems.contactmodel as contactmodel
from configuration.build_from_config import build_from_config
from configuration.parametricmodel import (
    ConstantModelConfig,
    FlatModelConfig,
    PiecewiseModelConfig,
)
from pycito.systems.contactmodel import ConstantModel, FlatModel, PiecewiseModel


def test_build_constant_model() -> None:
    config = ConstantModelConfig(const=1.0)
    model = build_from_config(contactmodel, config)
    assert isinstance(model, ConstantModel)


def test_build_flat_model() -> None:
    config = FlatModelConfig(location=0.0, direction=(0, 0, 1.0))
    model = build_from_config(contactmodel, config)
    assert isinstance(model, FlatModel)


def test_build_piecewise_model() -> None:
    config = PiecewiseModelConfig(
        breaks=[0, 1],
        models=[
            ConstantModelConfig(const=0),
            FlatModelConfig(location=1, direction=(0, 0, 1.0)),
            ConstantModelConfig(const=1),
        ],
    )
    model = build_from_config(contactmodel, config)
    assert isinstance(model, PiecewiseModel)
    assert isinstance(model._models[0], ConstantModel)
    assert isinstance(model._models[1], FlatModel)
    assert isinstance(model._models[2], ConstantModel)
    assert len(model._models) == 3
