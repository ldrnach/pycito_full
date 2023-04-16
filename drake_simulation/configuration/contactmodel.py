from dataclasses import dataclass
from typing import Literal, Union

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


@dataclass
class SemiparametricContactModelWithAmbiguityConfig:
    surface: SemiparametricModelConfig
    friction: SemiparametricModelConfig
    type: Literal[
        "SemiparametricContactModelWithAmbiguity"
    ] = "SemiparametricContactModelWithAmbiguity"


ContactModelConfig = Union[
    ContactModelConfig,
    SemiparametricContactModelConfig,
    SemiparametricContactModelWithAmbiguityConfig,
]
