from dataclasses import dataclass
from typing import Tuple, List, Union, Literal


@dataclass
class FlatModelConfig:
    location: float
    direction: Tuple[float, float, float]
    type: Literal["FlatModel"] = "FlatModel"


@dataclass
class ConstantModelConfig:
    const: float
    type: Literal["ConstantModel"] = "ConstantModel"


@dataclass
class PiecewiseModelConfig:
    breaks: List[float]
    models: List[Union[FlatModelConfig, ConstantModelConfig]]
    type: Literal["PiecewiseModel"] = "PiecewiseModel"


ParametricModelConfig = Union[
    FlatModelConfig, ConstantModelConfig, PiecewiseModelConfig
]
