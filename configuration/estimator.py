from dataclasses import dataclass

from configuration.kernel import WhiteNoiseKernelConfig
from configuration.parametricmodel import ConstantModelConfig, FlatModelConfig
from configuration.semiparametricmodel import SemiparametricModelConfig

from .contactmodel import ContactConfig, SemiparametricContactModelConfig
from .lcptype import LCP
from .optimization import SNOPTConfig


@dataclass
class EstimatorCostConfig:
    force: float = 1e0
    distance: float = 1
    friciton: float = 1
    velocity_scale: float = 1e-3
    force_scale: float = 1e2
    relaxation: float = 1e3


@dataclass
class EstimatorConfig:
    horizon: int = 1
    cost: EstimatorCostConfig = EstimatorCostConfig()
    contact_model: ContactConfig = SemiparametricContactModelConfig(
        surface=SemiparametricModelConfig(
            prior=FlatModelConfig(location=0, direction=[0, 0, 1]),
            kernel=WhiteNoiseKernelConfig(noise=1),
        ),
        friction=SemiparametricModelConfig(
            prior=ConstantModelConfig(const=0.0), kernel=WhiteNoiseKernelConfig(noise=1)
        ),
    )
    lcp: LCP = "ConstantRelaxedPseudoLinearComplementarityConstraint"
    solver: SNOPTConfig = SNOPTConfig(
        major_feasibility_tolerance=1e-6, major_optimality_tolerance=1e-6
    )
