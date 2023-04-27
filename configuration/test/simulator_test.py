import os
from unittest.mock import Mock, patch

from configuration.build_from_config import build_from_config
from configuration.controller import MPCControllerConfig
from configuration.simulator import DrakeSimulatorConfig
from drake_simulation import a1_simulator

REFERENCE = os.path.join("data", "a1", "reference", "symmetric", "3m", "reftraj.pkl")
URDF = os.path.join(
    "pycito", "systems", "A1", "A1_description", "urdf", "a1_foot_collision.urdf"
)


def test_build_simulator() -> None:
    controller_config = MPCControllerConfig(
        timestep=0.01, reference_path=REFERENCE, horizon=5
    )
    config = DrakeSimulatorConfig(
        timestep=0.01,
        urdf=URDF,
        environment="FlatGroundEnvironment",
        controller=controller_config,
    )
    with patch.object(
        a1_simulator.A1DrakeSimulationBuilder, "_addVisualizer", Mock(return_value=None)
    ):
        simulator = build_from_config(a1_simulator, config)
        assert isinstance(simulator, a1_simulator.A1DrakeSimulationBuilder)
