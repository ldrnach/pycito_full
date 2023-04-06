from dataclasses import dataclass

from .optimization import SNOPTConfig
from .contactmodel import ContactModelConfig


@dataclass
class EstimatorCostConfig:
    force: float
    distance: float
    friciton: float
    velocity_scale: float
    force_scale: float
    relaxation: float


@dataclass
class EstimatorConfig:
    horizon: int
    cost: EstimatorCostConfig
    solver: SNOPTConfig
    contact_model: ContactModelConfig
