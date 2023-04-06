from dataclasses import dataclass
from typing import Union, List, Literal

from .optimization import SNOPTConfig
from .estimator import EstimatorConfig
from .lcptype import LCP


@dataclass
class MPCCostConfig:
    position: float
    velocity: float
    control: float
    force: float
    slack: float
    jlimit: float
    complementarity_schedule: List[float]


# TODO: Reference_path is not well-defined
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
class ContactEILControllerConfig(MPCControllerConfig):
    estimator: EstimatorConfig
    type: Literal["A1ContactEILController"] = "A1ContactEILController"


ControllerConfig = Union[MPCControllerConfig, ContactEILControllerConfig]
