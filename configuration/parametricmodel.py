from dataclasses import dataclass
from typing import List, Literal, Tuple, Union


@dataclass
class FlatModelConfig:
    location: float
    direction: List[float]
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
