from dataclasses import dataclass
from typing import Union, List, Literal

from drake_simulation.configuration.optimization import SNOPTConfig


@dataclass
class MPCCostConfig:
    position: float
    velocity: float
    control: float
    force: float
    slack: float
    jlimit: float
    complementarity_schedule: List[float]


@dataclass
class MPCControllerConfig:
    timestep: float
    reference_path: str
    horizon: int
    lcptype: str
    cost: MPCCostConfig
    solver_config: SNOPTConfig
    type: Literal["A1ContactMPCController"] = "A1ContactMPCController"


@dataclass
class ContactEILControllerConfig(MPCControllerConfig):
    type: Literal["A1ContactEILController"] = "A1ContactEILController"


ControllerConfig = Union[MPCControllerConfig, ContactEILControllerConfig]
