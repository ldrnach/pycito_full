from dataclasses import dataclass
from typing import List, Literal, Union

from matplotlib import use

from .estimator import EstimatorConfig
from .lcptype import LCP
from .optimization import SNOPTConfig


@dataclass
class MPCCostConfig:
    base_position: float = 1e2
    joint_position: float = 1e2
    velocity: float = 1e-2
    control: float = 1e-3
    force: float = 0
    slack: float = 0
    jlimit: float = 0
    complementarity_schedule: List[float] = [1e-2, 1e-4]


@dataclass
class MPCControllerConfig:
    timestep: float
    reference_path: str
    horizon: int = 5
    lcptype: LCP = "ConstantRelaxedPseudoLinearComplementarityConstraint"
    cost: MPCCostConfig = MPCCostConfig()
    solver_config: SNOPTConfig = SNOPTConfig(
        major_feasibility_tolerance=1e-5,
        major_optimality_tolerance=1e-5,
        scale_option=0,
        major_step_limit=2.0,
        superbasics_limit=1000,
        linesearch_tolerance=0.9,
        iterations_limit=10000,
        use_basis_file=True,
    )
    type: Literal["A1ContactMPCController"] = "A1ContactMPCController"


@dataclass
class ContactEILControllerConfig:
    timestep: float
    reference_path: str
    horizon: int = 5
    lcptype: LCP = "ConstantRelaxedPseudoLinearComplementarityConstraint"
    cost: MPCCostConfig = MPCCostConfig()
    solver_config: SNOPTConfig = SNOPTConfig(
        major_feasibility_tolerance=1e-5,
        major_optimality_tolerance=1e-5,
        scale_option=0,
        major_step_limit=2.0,
        superbasics_limit=1000,
        linesearch_tolerance=0.9,
        iterations_limit=10000,
        use_basis_file=True,
    )
    estimator: EstimatorConfig = EstimatorConfig()
    type: Literal["A1ContactEILController"] = "A1ContactEILController"


ControllerConfig = Union[MPCControllerConfig, ContactEILControllerConfig]
