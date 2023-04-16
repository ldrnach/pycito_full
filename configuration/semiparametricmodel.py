from dataclasses import dataclass
from typing import Literal

from .kernel import KernelConfig
from .parametricmodel import ParametricModelConfig


@dataclass
class SemiparametricModelConfig:
    prior: ParametricModelConfig
    kernel: KernelConfig
    type: Literal["SemiparametricModel"] = "SemiparametricModel"
