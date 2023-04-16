from ast import Constant
from re import L

import pycito.systems.kernels as kernels
from configuration.build_from_config import build_from_config
from configuration.kernel import (
    CenteredLinearKernelConfig,
    ConstantKernelConfig,
    HyperbolicTangentKernelConfig,
    LinearKernelConfig,
    PolynomialKernelConfig,
    PseudoHuberKernelConfig,
    RBFKernelConfig,
    RegularizedCenteredLinearKernelConfig,
    RegularizedConstantKernelConfig,
    RegularizedHyperbolicKernelConfig,
    RegularizedLinearKernelConfig,
    RegularizedPolynomialKernelConfig,
    RegularizedPseudoHuberKernelConfig,
    RegularizedRBFKernelConfig,
    WhiteNoiseKernelConfig,
)
from pycito.systems.kernels import (
    CenteredLinearKernel,
    ConstantKernel,
    HyperbolicTangentKernel,
    LinearKernel,
    PolynomialKernel,
    PseudoHuberKernel,
    RBFKernel,
    RegularizedCenteredLinearKernel,
    RegularizedConstantKernel,
    RegularizedHyperbolicKernel,
    RegularizedLinearKernel,
    RegularizedPolynomialKernel,
    RegularizedPseudoHuberKernel,
    RegularizedRBFKernel,
    WhiteNoiseKernel,
)
from pycito.tests.unittests.kernels_test import HyperbolicKernelTest


def test_build_rbf_kernel() -> None:
    config_1d = RBFKernelConfig(length_scale=[1.0])
    kernel = build_from_config(kernels, config_1d)
    assert isinstance(kernel, RBFKernel)

    config_3d = RBFKernelConfig(length_scale=[1.0, 1.0, 1.0])
    kernel = build_from_config(kernels, config_3d)
    assert isinstance(kernel, RBFKernel)


def test_build_pseudo_huber() -> None:
    config = PseudoHuberKernelConfig(length_scale=[1.0, 1.0, 2.0], delta=0.1)
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, PseudoHuberKernel)


def test_build_linear_kernel() -> None:
    config = LinearKernelConfig(weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], offset=0.1)
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, LinearKernel)


def test_build_centered_linear_kernel() -> None:
    config = CenteredLinearKernelConfig(weights=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, CenteredLinearKernel)


def test_build_hyberbolic_tangent_kernel() -> None:
    config = HyperbolicTangentKernelConfig(
        weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], offset=0.1
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, HyperbolicTangentKernel)


def test_build_polynomial_kernel() -> None:
    config = PolynomialKernelConfig(
        weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], offset=0.1, degree=2
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, PolynomialKernel)


def test_build_constant_kernel() -> None:
    config = ConstantKernelConfig(const=1)
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, ConstantKernel)


def test_build_white_noise_kernel() -> None:
    config = WhiteNoiseKernelConfig(noise=1.0)
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, WhiteNoiseKernel)


def test_build_regularized_rbf_kernel() -> None:
    config = RegularizedRBFKernelConfig(length_scale=[1, 1, 1], noise=0.1)
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedRBFKernel)


def test_build_regularized_pseudo_huber_kernel() -> None:
    config = RegularizedPseudoHuberKernelConfig(
        length_scale=[1, 1, 1], delta=0.1, noise=0.01
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedPseudoHuberKernel)


def test_build_regularized_linear_kernel_config() -> None:
    config = RegularizedLinearKernelConfig(
        weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], offset=0.1, noise=0.01
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedLinearKernel)


def test_build_regularized_hyperbolic_kernel() -> None:
    config = RegularizedHyperbolicKernelConfig(
        weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], offset=0.1, noise=0.01
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedHyperbolicKernel)


def test_build_regularized_polynomial_kernel() -> None:
    config = RegularizedPolynomialKernelConfig(
        weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], offset=0.1, degree=2, noise=0.01
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedPolynomialKernel)


def test_build_regularized_constant_kernel_config() -> None:
    config = RegularizedConstantKernelConfig(const=1, noise=0.01)
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedConstantKernel)


def test_build_regularized_centered_linear_kernel() -> None:
    config = RegularizedCenteredLinearKernelConfig(
        weights=[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], noise=0.1
    )
    kernel = build_from_config(kernels, config)
    assert isinstance(kernel, RegularizedCenteredLinearKernel)
