from dataclasses import dataclass
from typing import List, Literal, Union


@dataclass
class RBFKernelConfig:
    length_scale: List[float]
    type: Literal["RBFKernel"] = "RBFKernel"


@dataclass
class PseudoHuberKernelConfig:
    length_scale: List[float]
    delta: float
    type: Literal["PsuedoHuberKernel"] = "PseudoHuberKernel"


@dataclass
class LinearKernelBase:
    weights: List[List[float]]
    offset: float


@dataclass
class LinearKernelConfig(LinearKernelBase):
    type: Literal["LinearKernel"] = "LinearKernel"


@dataclass
class CenteredLinearKernelConfig(LinearKernelBase):
    weights: List[List[float]]
    type: Literal["CenteredLinearKernel"] = "CenteredLinearKernel"


@dataclass
class HyperbolicTangentKernelConfig(LinearKernelBase):
    type: Literal["HyperbolicTangentKernel"] = "HyperbolicTangentKernel"


@dataclass
class PolynomialKernelConfig(LinearKernelBase):
    degree: float
    type: Literal["PolynomialKernel"] = "PolynomialKernel"


@dataclass
class ConstantKernelConfig:
    const: float
    type: Literal["ConstantKernel"] = "ConstantKernel"


@dataclass
class WhiteNoiseKernelConfig:
    noise: float
    type: Literal["WhiteNoiseKernel"] = "WhiteNoiseKernel"


@dataclass
class RegularizedRBFKernelConfig:
    length_scale: List[float]
    noise: float
    type: Literal["RegularizedRBFKernel"] = "RegularizedRBFKernel"


@dataclass
class RegularizedPsuedoHuberKernelConfig:
    length_scale: List[float]
    delta: float
    noise: float
    type: Literal["RegularizedPseudoHuberKernel"] = "RegularizedPseudoHuberKernel"


@dataclass
class RegularizedLinearKernelConfig(LinearKernelBase):
    noise: float
    type: Literal["RegularizedLinearKernel"] = "RegularizedLinearKernel"


@dataclass
class RegularizedHyberbolicKernelConfig(LinearKernelBase):
    noise: float
    type: Literal["RegularizedHyperbolicKernel"] = "RegularizedHyperbolicKernel"


@dataclass
class RegularizedPolynomialKernelConfig(LinearKernelBase):
    degree: float
    noise: float
    type: Literal["RegularizedPolynomialKernel"] = "RegularizedPolynomialKernel"


@dataclass
class RegularizedConstantKernelConfig:
    const: float
    noise: float
    type: Literal["RegularizedConstantKernelConfig"] = "RegularizedConstantKernelConfig"


@dataclass
class RegularizedCenteredLinearKernelConfig:
    weights: List[List[float]]
    noise: float
    type: Literal["RegularizedCenteredLinearKernel"] = "RegularizedCenteredLinearKernel"


KernelConfig = Union[
    RBFKernelConfig,
    PseudoHuberKernelConfig,
    PseudoHuberKernelConfig,
    LinearKernelConfig,
    CenteredLinearKernelConfig,
    HyperbolicTangentKernelConfig,
    PolynomialKernelConfig,
    ConstantKernelConfig,
    WhiteNoiseKernelConfig,
    RegularizedRBFKernelConfig,
    RegularizedPsuedoHuberKernelConfig,
    RegularizedLinearKernelConfig,
    RegularizedHyberbolicKernelConfig,
    RegularizedPolynomialKernelConfig,
    RegularizedConstantKernelConfig,
    RegularizedCenteredLinearKernelConfig,
]
