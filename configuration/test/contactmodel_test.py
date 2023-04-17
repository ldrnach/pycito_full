from configuration import (
    ConstantKernelConfig,
    ConstantModelConfig,
    ContactModelConfig,
    SemiparametricContactModelConfig,
    SemiparametricContactModelWithAmbiguityConfig,
)
from configuration.build_from_config import build_from_config
from configuration.parametricmodel import FlatModelConfig
from configuration.semiparametricmodel import SemiparametricModelConfig
from pycito.systems import contactmodel
from pycito.systems.contactmodel import (
    ContactModel,
    SemiparametricContactModel,
    SemiparametricContactModelWithAmbiguity,
)
from pycito.systems.contactmodel.parametric_model import ConstantModel, FlatModel
from pycito.systems.contactmodel.semiparametric_model import SemiparametricModel
from pycito.systems.kernels import ConstantKernel


def test_build_contact_model() -> None:
    surface = ConstantModelConfig(const=1.0)
    friction = ConstantModelConfig(const=1.0)
    config = ContactModelConfig(surface=surface, friction=friction)
    model = build_from_config(contactmodel, config)
    assert isinstance(model, ContactModel)
    assert isinstance(model.surface, contactmodel.ConstantModel)
    assert isinstance(model.friction, contactmodel.ConstantModel)


def test_build_semiparametric_contact_model() -> None:
    surface = SemiparametricModelConfig(
        prior=FlatModelConfig(location=0.1, direction=(0, 0, 1.0)),
        kernel=ConstantKernelConfig(const=1.0),
    )
    friction = SemiparametricModelConfig(
        prior=ConstantModelConfig(const=1.0), kernel=ConstantKernelConfig(const=1.0)
    )
    config = SemiparametricContactModelConfig(surface, friction)
    model = build_from_config(contactmodel, config)
    assert isinstance(model, SemiparametricContactModel)
    assert isinstance(model.surface, SemiparametricModel)
    assert isinstance(model.friction, SemiparametricModel)
    assert isinstance(model.surface.prior, FlatModel)
    assert isinstance(model.surface.kernel, ConstantKernel)
    assert isinstance(model.friction.prior, ConstantModel)
    assert isinstance(model.friction.kernel, ConstantKernel)


def test_build_semiparametric_with_ambiguity_model() -> None:
    surface = SemiparametricModelConfig(
        prior=FlatModelConfig(location=0.1, direction=(0, 0, 1.0)),
        kernel=ConstantKernelConfig(const=1.0),
    )
    friction = SemiparametricModelConfig(
        prior=ConstantModelConfig(const=1.0), kernel=ConstantKernelConfig(const=1.0)
    )
    config = SemiparametricContactModelWithAmbiguityConfig(surface, friction)
    model = build_from_config(contactmodel, config)
    assert isinstance(model, SemiparametricContactModelWithAmbiguity)
    assert isinstance(model.surface, SemiparametricModel)
    assert isinstance(model.friction, SemiparametricModel)
    assert isinstance(model.surface.prior, FlatModel)
    assert isinstance(model.surface.kernel, ConstantKernel)
    assert isinstance(model.friction.prior, ConstantModel)
    assert isinstance(model.friction.kernel, ConstantKernel)
