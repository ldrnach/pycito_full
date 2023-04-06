from dataclasses import dataclass
from typing import Literal

from .parametricmodel import ParametricModelConfig
from .kernel import KernelConfig


@dataclass
class SemiparametricModelConfig:
    type: Literal["SemiparametricModel"] = "SemiparametricModel"
    prior: ParametricModelConfig
    kernel: KernelConfig
