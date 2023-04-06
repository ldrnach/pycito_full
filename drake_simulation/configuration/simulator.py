from dataclasses import dataclass
from typing import Literal
from drake_simulation.configuration.controller import ControllerConfig


@dataclass
class DrakeSimulatorConfig:
    timestep: float
    urdf: str
    environment: Literal[
        "FlatGroundEnvironment", "FlatGroundWithFrictionPatch", "RampUpEnvironment"
    ]
    controller: ControllerConfig
