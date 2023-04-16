from dataclasses import dataclass
from typing import Literal

from .kernel import KernelConfig
from .parametricmodel import ParametricModelConfig


@dataclass
class SemiparametricModelConfig:
    type: Literal["SemiparametricModel"] = "SemiparametricModel"
    prior: ParametricModelConfig
    kernel: KernelConfig
