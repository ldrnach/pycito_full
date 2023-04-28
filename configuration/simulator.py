from dataclasses import dataclass
from typing import Literal

from configuration.controller import ControllerConfig

from .file_constants import URDF


@dataclass
class DrakeSimulatorConfig:
    timestep: float
    environment: Literal[
        "FlatGroundEnvironment", "FlatGroundWithFrictionPatch", "RampUpEnvironment"
    ]
    controller: ControllerConfig
    urdf: str = str(URDF)
    type: Literal["A1DrakeSimulationBuilder"] = "A1DrakeSimulationBuilder"
