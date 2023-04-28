from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Union

from .estimator import EstimatorConfig
from .file_constants import FLAT_GROUND_REFERENCE
from .lcptype import LCP
from .optimization import SNOPTConfig


@dataclass
class MPCCostConfig:
    base_position: float
    joint_position: float
    velocity: float
    control: float
    force: float
    slack: float
    jlimit: float
    complementarity_schedule: List[float] = field(default_factory=list)

    @classmethod
    def default(cls) -> MPCCostConfig:
        return cls(
            base_position=1e2,
            joint_position=1e2,
            velocity=1e-2,
            control=1e-3,
            force=0,
            slack=0,
            jlimit=0,
            complementarity_schedule=[1e-2, 1e-4],
        )


@dataclass
class MPCControllerConfig:
    timestep: float
    reference_path: str = field(default_factory=lambda: str(FLAT_GROUND_REFERENCE))
    horizon: int = 5
    lcptype: LCP = "ConstantRelaxedPseudoLinearComplementarityConstraint"
    cost: MPCCostConfig = field(default_factory=MPCCostConfig.default)
    solver_config: SNOPTConfig = field(
        default=SNOPTConfig(
            major_feasibility_tolerance=1e-5,
            major_optimality_tolerance=1e-5,
            scale_option=0,
            major_step_limit=2.0,
            superbasics_limit=1000,
            linesearch_tolerance=0.9,
            iterations_limit=10000,
            use_basis_file=True,
        )
    )

    type: Literal["A1ContactMPCController"] = "A1ContactMPCController"


@dataclass
class ContactEILControllerConfig:
    timestep: float
    reference_path: str = field(default_factory=lambda: str(FLAT_GROUND_REFERENCE))
    horizon: int = 5
    lcptype: LCP = "ConstantRelaxedPseudoLinearComplementarityConstraint"
    cost: MPCCostConfig = field(default_factory=MPCCostConfig.default)
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
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    type: Literal["A1ContactEILController"] = "A1ContactEILController"


ControllerConfig = Union[MPCControllerConfig, ContactEILControllerConfig]
