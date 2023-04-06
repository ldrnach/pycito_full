from dataclasses import dataclass
from typing import Union, Literal

from .parametricmodel import ParametricModelConfig
from .semiparametricmodel import SemiparametricModelConfig


@dataclass
class ContactModelConfig:
    surface: ParametricModelConfig
    friction: ParametricModelConfig
    type: Literal["ContactModel"] = "ContactModel"


@dataclass
class SemiparametricContactModelConfig:
    surface: SemiparametricModelConfig
    friction: SemiparametricModelConfig
    type: Literal["SemiparametricContactModel"] = "SemiparametricContactModel"


ContactModelConfig = Union[ContactModelConfig, SemiparametricContactModelConfig]
