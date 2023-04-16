from __future__ import annotations

import abc
import warnings

import matplotlib.pyplot as plt
import numpy as np
from parametric_model import ConstantModel, DifferentiableModel, FlatModel
from pydrake.all import MathematicalProgram, Solve

import pycito.decorators as deco
from drake_simulation.configuration.build_from_config import build_from_config
from drake_simulation.configuration.contactmodel import ContactModelConfig

from . import parametric_model


def householderortho3D(normal):
    """
    Use 3D Householder vector orthogonalization

    Arguments:
        normal: (3,) numpy array, the normal vector
    Returns:
        tangent: (3,) numpy array, orthogonal to normal
        binormal; (3,) numpy array, orthogonal to tangent and normal
    """
    if normal[0] < 0 and normal[1] == 0 and normal[2] == 0:
        normal = -normal
    mag = np.linalg.norm(normal)
    h = np.zeros_like(normal)
    h[0] = max([normal[0] - mag, normal[0] + mag])
    h[1:] = np.copy(normal[1:])
    # Calculate the tangent and binormal vectors
    hmag = np.sum(h**2)
    tangent = np.array(
        [-2 * h[0] * h[1] / hmag, 1 - 2 * h[1] ** 2 / hmag, -2 * h[1] * h[2] / hmag]
    )
    binormal = -np.array(
        [-2 * h[0] * h[2] / hmag, -2 * h[1] * h[2] / hmag, 1 - 2 * h[2] ** 2 / hmag]
    )
    return binormal, tangent


class _ContactModel(abc.ABC):
    """
    Abstract base class for specifying a generic contact model
    """

    def str(self):
        return f"{type(self).__name__}"

    @abc.abstractclassmethod
    def eval_surface(self, pt):
        """
        Returns the value of the level sets of the surface geometry at the supplied point

        If eval_surface(pt) > 0, then the point is not in contact with the surface
        If eval_surface(pt) = 0, then the point is on the surface
        If eval_surface(pt) < 0, then the point is inside the surface

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return Values
            out: a (1,) numpy array, the surface evaluation (roughly the 'distance')
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def eval_friction(self, pt):
        """
        Returns the value of the level sets of the surface friction at the supplied point. Note that the values of friction returned by eval_friction may only be considered accurate when eval_surface(pt) = 0

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return Values
            out: a (1,) numpy array, the friction coefficient evaluation
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def local_frame(self, pt):
        """
        Return the local coordinate frame of the surface geometry at the specified point.

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return values:
            R: a (3,3) numpy array. The first row is the surface normal vector. The next two rows are the surface tangent vectors
        """
        raise NotImplementedError

    def find_surface_zaxis_zeros(self, pts):
        # Setup the mathematical program
        soln = np.copy(pts)
        guess = pts[2:, 0]
        for k, pt in enumerate(pts.transpose()):
            prog = MathematicalProgram()
            zvar = prog.NewContinuousVariables(rows=1, name="z")
            # Find the smallest modification
            prog.AddQuadraticErrorCost(np.eye(1), pt[2:], vars=zvar)
            # Constrain the surface to evaluate to zero
            pt_cstr = lambda z, x=pt[0], y=pt[1]: self.eval_surface(
                np.concatenate([np.array([x, y]), z], axis=0)
            )
            prog.AddConstraint(
                pt_cstr,
                lb=np.zeros((1,)),
                ub=np.zeros((1,)),
                vars=zvar,
                description="zeroset",
            )
            # Solve the program
            prog.SetInitialGuess(zvar, guess)
            result = Solve(prog)
            if not result.is_success():
                warnings.warn(
                    f"find_surface_zaxis_zeros did not solve successfully. Results may be inaccurate"
                )
            soln[2, k] = result.GetSolution(zvar)
            guess = soln[2:, k]
        return soln

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
        # Make the plots
        axs[0].plot(surf_pts[0], surf_pts[2], linewidth=1.5, label=label)
        axs[0].set_ylabel("Contact Height (m)")
        axs[1].plot(pts[0, :], fric_pts, linewidth=1.5, label=label)
        axs[1].set_ylabel("Friction Coefficient")
        axs[1].set_xlabel("Position (m)")
        return fig, axs


class ContactModel(_ContactModel):
    def __init__(self, surface, friction):
        assert issubclass(
            type(surface), DifferentiableModel
        ), "surface must be a subclass of DifferentiableModel"
        assert issubclass(
            type(friction), DifferentiableModel
        ), "friction must be a subclass of DifferentiableModel"
        self.surface = surface
        self.friction = friction

    @classmethod
    def build_from_config(cls, config: ContactModelConfig) -> ContactModel:
        surface = build_from_config(parametric_model, config.surface)
        friction = build_from_config(parametric_model, config.friction)
        return cls(surface, friction)

    @classmethod
    def FlatSurfaceWithConstantFriction(
        cls, location=0.0, friction=1.0, direction=np.array([0.0, 0.0, 1.0])
    ):
        """
        Create a contact model using a flat surface with constant friction
        """
        surf = FlatModel(location, direction)
        fric = ConstantModel(friction)
        return cls(surf, fric)

    def eval_surface(self, pt):
        """
        Returns the value of the level sets of the surface geometry at the supplied point

        If eval_surface(pt) > 0, then the point is not in contact with the surface
        If eval_surface(pt) = 0, then the point is on the surface
        If eval_surface(pt) < 0, then the point is inside the surface

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return Values
            out: a (1,) numpy array, the surface evaluation (roughly the 'distance')
        """
        return self.surface(pt)

    def eval_friction(self, pt):
        """
        Returns the value of the level sets of the surface friction at the supplied point. Note that the values of friction returned by eval_friction may only be considered accurate when eval_surface(pt) = 0

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return Values
            out: a (1,) numpy array, the friction coefficient evaluation
        """
        return self.friction(pt)

    def local_frame(self, pt):
        """
        Return the local coordinate frame of the surface geometry at the specified point.

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return values:
            R: a (3,3) numpy array. The first row is the surface normal vector. The next two rows are the surface tangent vectors
        """
        normal = self.surface.gradient(pt).flatten()
        normal = normal / np.linalg.norm(normal)
        tangent, binormal = householderortho3D(normal)
        return np.row_stack([normal, tangent, binormal])
