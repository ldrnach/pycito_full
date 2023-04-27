from dataclasses import dataclass
from typing import Literal

from configuration.controller import ControllerConfig


@dataclass
class DrakeSimulatorConfig:
    timestep: float
    urdf: str
    environment: Literal[
        "FlatGroundEnvironment", "FlatGroundWithFrictionPatch", "RampUpEnvironment"
    ]
    controller: ControllerConfig
    type: Literal["A1DrakeSimulationBuilder"] = "A1DrakeSimulationBuilder"
