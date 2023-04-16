from __future__ import annotations

import copy

import numpy as np
from semiparametric_model import SemiparametricModel

from configuration.build_from_config import build_from_config
from configuration.contactmodel import (
    SemiparametricContactModelConfig,
    SemiparametricContactModelWithAmbiguityConfig,
)

from . import semiparametric_model
from .contact_model import ContactModel


class SemiparametricContactModel(ContactModel):
    def __init__(self, surface, friction):
        assert isinstance(
            surface, SemiparametricModel
        ), "surface must be a semiparametric model"
        assert isinstance(
            friction, SemiparametricModel
        ), "friction must be a semiparametric model"
        super(SemiparametricContactModel, self).__init__(surface, friction)

    @classmethod
    def build_from_config(
        cls, config: SemiparametricContactModelConfig
    ) -> SemiparametricContactModel:
        surface = build_from_config(semiparametric_model, config.surface)
        friction = build_from_config(semiparametric_model, config.friction)
        return cls(surface, friction)

    @classmethod
    def FlatSurfaceWithRBFKernel(
        cls, height=0.0, friction=1.0, length_scale=0.1, reg=0.0
    ):
        """
        Factory method for constructing a semiparametric contact model

        Assumes the prior is a flat surface with constant friction
        uses independent RBF kernels for the surface and friction models
        """
        surf = SemiparametricModel.FlatPriorWithRBFKernel(
            location=height, length_scale=length_scale, reg=reg
        )
        fric = SemiparametricModel.ConstantPriorWithRBFKernel(
            const=friction, length_scale=length_scale, reg=reg
        )
        return cls(surf, fric)

    @classmethod
    def FlatSurfaceWithHuberKernel(
        cls, height=0.0, friction=1.0, length_scale=0.1, delta=0.1, reg=0.0
    ):
        """
        Factory method for constructing a semiparametric contact model
        Assumes the prior is a flat surface with constant friction
        Uses independent Pseudo-Huber kernels for the surface and friction models
        """
        surf = SemiparametricModel.FlatPriorWithHuberKernel(
            location=height, length_scale=length_scale, delta=delta, reg=reg
        )
        fric = SemiparametricModel.ConstantPriorWithHuberKernel(
            const=friction, length_scale=length_scale, delta=delta, reg=reg
        )

        return cls(surf, fric)

    @classmethod
    def RBFSurfaceWithHuberFriction(
        cls,
        height=0.0,
        friction=1.0,
        height_length=0.1,
        friction_length=0.1,
        delta=0.1,
        reg=0.0,
    ):
        """
        Factory method for constructing a semiparametric contact model
        Assumes the prior is a flat surface with constant friction
        Uses a RBF kernel for the surface model and a pseudo-huber kernel for the friction model

        Arguments:
            height (float): the height of the flat surface prior
            friction (float): the value of the constant friction prior
            height_length (float): the length scale value for the surface RBF kernel
            friction_length (float): the length scale value for the friction PseudoHuber kernel
            delta (float): the delta value for the friction PseudoHuber kernel
        """
        surf = SemiparametricModel.FlatPriorWithRBFKernel(
            location=height, length_scale=height_length, reg=reg
        )
        fric = SemiparametricModel.ConstantPriorWithHuberKernel(
            const=friction, length_scale=friction_length, delta=delta, reg=reg
        )
        return cls(surf, fric)

    def add_samples(self, sample_points, surface_weights, friction_weights):
        """
        Add samples to the semiparametric model

        """
        self.surface.add_samples(sample_points, surface_weights)
        self.friction.add_samples(sample_points, friction_weights)

    def get_sample_points(self):
        return self.surface._sample_points

    def get_surface_weights(self):
        return self.surface._kernel_weights

    def get_friction_weights(self):
        return self.friction._kernel_weights

    def compress(self, atol=1e-4):
        self.surface.compress(atol)
        self.friction.compress(atol)

    @property
    def surface_kernel(self):
        return self.surface.kernel

    @property
    def friction_kernel(self):
        return self.friction.kernel

    def toSemiparametricModelWithAmbiguity(self):
        """
        Upcast model to SemiparametricContactModelWithAmbiguity
        """
        model = SemiparametricContactModelWithAmbiguity(self.surface, self.friction)
        if model.get_sample_points() is not None:
            model.add_samples(
                model.get_sample_points(),
                model.get_surface_weights(),
                model.get_friction_weights(),
            )
        return model


class SemiparametricContactModelWithAmbiguity(SemiparametricContactModel):
    def __init__(self, surface, friction):
        super().__init__(surface, friction)
        self.lower_bound = SemiparametricContactModel(
            copy.deepcopy(surface), copy.deepcopy(friction)
        )
        self.upper_bound = SemiparametricContactModel(
            copy.deepcopy(surface), copy.deepcopy(friction)
        )

    @classmethod
    def build_from_config(
        cls, config: SemiparametricContactModelWithAmbiguityConfig
    ) -> SemiparametricContactModelWithAmbiguity:
        surface = build_from_config(semiparametric_model, config.surface)
        friction = build_from_config(semiparametric_model, config.friction)
        return cls(surface, friction)

    def add_samples(self, sample_points, surface_weights, friction_weights):
        """
        Add samples to the semiparametric model

        """
        super().add_samples(sample_points, surface_weights, friction_weights)
        self.lower_bound.add_samples(sample_points, surface_weights, friction_weights)
        self.upper_bound.add_samples(sample_points, surface_weights, friction_weights)

    def set_lower_bound(self, surface_weights, friction_weights):
        """
        Set the weights in the lower bound model

        """
        self.lower_bound.surface._kernel_weights = surface_weights
        self.lower_bound.friction._kernel_weights = friction_weights

    def set_upper_bound(self, surface_weights, friction_weights):
        """
        Set the weights in the upper bound model

        """
        self.upper_bound.surface._kernel_weights = surface_weights
        self.upper_bound.friction._kernel_weights = friction_weights

    def toSemiparametricModel(self):
        return SemiparametricContactModel(self.surface, self.friction)

    @deco.showable_fig
    @deco.saveable_fig
    def plot2D(self, pts, axs=None, label=None):
        """
        Plot the contact model in 2D coordinates. Currently, plot2D requires a full 3D point specification, but plots the z-axis values of the terrain along the x-axis (the y-values are ignored).

        plot2D first finds the closest zeros of the contact surface model, and then plots the corresponding (x,z) value pair. plot2D also evaluates the friction coefficient at the given (x,y,z) triples

        Arguments:
            pts: (3,N) numpy array, points "close to" the contact model surface
        """
        # Get the figure and axis handles
        if axs is None:
            fig, axs = plt.subplots(2, 1)
        else:
            plt.sca(axs[0])
            fig = plt.gcf()
        # Evaluate the contact models
        surf_pts = self.find_surface_zaxis_zeros(pts)
        fric_pts = np.concatenate(
            [self.eval_friction(pt) for pt in pts.transpose()], axis=0
        )
        # Evaluate the upper and lower bound models
        # Note that the upper_bound model produces the greatest distance to the terrain, and is therefore the lower bound on the terrain location, and vice versa for the lower_bound model
        surf_ub = self.lower_bound.find_surface_zaxis_zeros(pts)
        fric_lb = np.concatenate(
            [self.lower_bound.eval_friction(pt) for pt in pts.transpose()], axis=0
        )
        surf_lb = self.upper_bound.find_surface_zaxis_zeros(pts)
        fric_ub = np.concatenate(
            [self.upper_bound.eval_friction(pt) for pt in pts.transpose()], axis=0
        )

        # Make the plots
        surf_line = axs[0].plot(surf_pts[0], surf_pts[2], linewidth=1.5, label=label)
        surf_limits = axs[0].get_ylim()
        axs[0].fill_between(
            surf_pts[0],
            surf_lb[2],
            surf_ub[2],
            alpha=0.2,
            color=surf_line[-1].get_color(),
        )
        axs[0].set_ylim(surf_limits)
        axs[0].set_ylabel("Contact Height (m)")

        fric_line = axs[1].plot(pts[0, :], fric_pts, linewidth=1.5, label=label)
        fric_limits = axs[1].get_ylim()
        axs[1].fill_between(
            pts[0, :], fric_lb, fric_ub, alpha=0.2, color=fric_line[-1].get_color()
        )
        axs[1].set_ylim(fric_limits)
        axs[1].set_ylabel("Friction Coefficient")
        axs[1].set_xlabel("Position (m)")
        return fig, axs
