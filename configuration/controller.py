from dataclasses import dataclass
from typing import List, Literal, Union

from .estimator import EstimatorConfig
from .lcptype import LCP
from .optimization import SNOPTConfig


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
    lcptype: LCP
    cost: MPCCostConfig
    solver_config: SNOPTConfig
    type: Literal["A1ContactMPCController"] = "A1ContactMPCController"


@dataclass
class ContactEILControllerConfig:
    timestep: float
    reference_path: str
    horizon: int
    lcptype: LCP
    cost: MPCCostConfig
    solver_config: SNOPTConfig
    estimator: EstimatorConfig
    type: Literal["A1ContactEILController"] = "A1ContactEILController"


ControllerConfig = Union[MPCControllerConfig, ContactEILControllerConfig]
