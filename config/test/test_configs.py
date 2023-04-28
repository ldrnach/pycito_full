from pathlib import Path

from configuration.contactmodel import SemiparametricContactModelConfig
from configuration.controller import ContactEILControllerConfig, MPCControllerConfig
from configuration.kernel import ConstantKernelConfig
from configuration.load_simulator_config import load_simulator_config
from configuration.semiparametricmodel import SemiparametricModelConfig
from configuration.simulator import DrakeSimulatorConfig


def test_example_mpc() -> None:
    file = Path("config", "examples", "example_mpc.json")
    config = load_simulator_config(file)
    assert isinstance(config, DrakeSimulatorConfig)
    assert isinstance(config.controller, MPCControllerConfig)


def test_example_campc() -> None:
    file = Path("config", "examples", "example_campc.json")
    config = load_simulator_config(file)
    assert isinstance(config, DrakeSimulatorConfig)
    assert isinstance(config.controller, ContactEILControllerConfig)
    assert isinstance(
        config.controller.estimator.contact_model, SemiparametricContactModelConfig
    )
    assert isinstance(
        config.controller.estimator.contact_model.friction, SemiparametricModelConfig
    )
    assert isinstance(
        config.controller.estimator.contact_model.friction.kernel, ConstantKernelConfig
    )
