"""
contactmodel.py: module for specifying arbitrary contact models for use with TimeSteppingRigidBodyPlant

Luke Drnach
February 16, 2022
"""

#TODO: rewrite models to always return numpy arrays
#TODO: Integrate with Timestepping
import numpy as np
import abc
import pycito.systems.kernels as kernels
from copy import deepcopy

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
    hmag = np.sum(h ** 2)
    tangent = np.array([-2 * h[0]*h[1]/hmag, 1 - 2 * h[1]**2 /hmag, -2*h[1]*h[2]/hmag])
    binormal = np.array([-2 *h[0] * h[2]/hmag, -2 * h[1] * h[2] / hmag, 1 - 2 * h[2]**2 / hmag])
    return tangent, binormal

class DifferentiableModel(abc.ABC):
    def __call__(self, x):
        return self.eval(x)

    @abc.abstractmethod
    def eval(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, x):
        raise NotImplementedError

class ConstantModel(DifferentiableModel):
    def __init__(self, const = 1.0):
        self._const = const

    def eval(self, x):    
        """
        Evalute the constant prior
        Note: the strange syntax ensures the prior works with autodiff types

        Arguments:
            x: (n, ) numpy array

        Return Values:
            out: (1, ) numpy array
        """
        return np.atleast_1d(np.vdot(np.zeros_like(x), x) + self._const)

    def gradient(self, x):
        """
        Evalute the gradient of the prior
        Note: the syntax 0*x ensures the prior works with autodiff types

        Arguments:
            x: (n, ) numpy array

        Return Values:
            out: (1, n) numpy array
        """
        return np.reshape(0 * x, (1, x.shape[0]))

class FlatModel(DifferentiableModel):
    def __init__(self, location = 1.0, direction = np.array([0, 0, 1])):
        self._location = location
        self._direction = direction

    def eval(self, x):
        """
        Evaluates the flat model, written to work with autodiff types

        Arguments:
            x: (3,) numpy array

        Return values:
            out: (1,) numpy array
        """
        return np.atleast_1d(np.vdot(self._direction, x) - self._location)

    def gradient(self, x):
        """
        Evaluate the gradient of the flat model, written to work with autodiff types

        Arguments:
            x: (3,) numpy array

        Return values:
            out: (1,3) numpy array
        """
        return np.reshape(self._direction + np.vdot(np.zeros_like(x), x), (1, self._direction.size))

class SemiparametricModel(DifferentiableModel):
    def __init__(self, prior, kernel):
        assert issubclass(type(prior), DifferentiableModel), "prior must be a concrete implementation of DifferentiablePrior"
        assert issubclass(type(kernel), kernels.DifferentiableStationaryKernel), "kernel must be a concrete implementation of DifferentiableStationaryKernel"
        self.prior = prior
        self.kernel = kernel
        self._kernel_weights = None
        self._sample_points = None

    @classmethod
    def ConstantPriorWithRBFKernel(cls, const=0, length_scale=1):
        return cls(prior = ConstantModel(const = const), kernel=kernels.RBFKernel(length_scale=length_scale))

    @classmethod
    def FlatPriorWithRBFKernel(cls, location = 0., direction = np.array([0, 0, 1]), length_scale = 1.):
        return cls(prior = FlatModel(location = location, direction = direction),
                    kernel = kernels.RBFKernel(length_scale=length_scale))

    @classmethod
    def ConstantPriorWithHuberKernel(cls, const=0, length_scale=1, delta=1):
        return cls(prior = ConstantModel(const = const),
                    kernel = kernels.PseudoHuberKernel(length_scale=length_scale, delta=delta))
    
    @classmethod
    def FlatPriorWithHuberKernel(cls, location = 0, direction = np.array([0, 0, 1]), length_scale = 1., delta = 1):
        return cls(prior = FlatModel(location, direction),
                    kernel = kernels.PseudoHuberKernel(length_scale, delta))

    def add_samples(self, samples, weights):
        """
        adds samples, and the relevant weights, to the model

        Arguments:
            samples (n_features, n_samples): numpy array of sample points
            weights (n_samples, ): numpy array of corresponding kernel weights
        """
        samples = np.atleast_2d(samples)
        assert samples.shape[1] == weights.shape[0], 'there must be as many weights as there are samples'
        if self._sample_points is not None:
            assert samples.shape[1] == self._sample_points.shape[1], 'new samples must have the same number of features as the existing samples'
            self._sample_points = np.concatenate([self._sample_points, samples], axis=0)
            self._kernel_weights = np.concatenate([self._kernel_weights, weights], axis=0)
        else:
            self._sample_points = samples
            self._kernel_weights = weights        

    def eval(self, x):
        """
        Evaluate the semiparametric model at the sample point
        
        Arguments:
            x: (n_features,) numpy array, the sample point at which to evaluate the model
        
        Return Values:
            y: (1,) the output of the model
        """
        # Evaluate the prior
        y = self.prior(x)
        # Evaluate the kernel
        if self._sample_points is not None:
            y = y + self.kernel(x, self._sample_points).dot(self._kernel_weights)
        return y

    def gradient(self, x):
        """
        Evaluate the gradient of the semiparametric model

        Arguments:
            x: (n_features,) numpy array, the sample point at which to evaluate the model

        Return value:
            grad (1, n_features) numpy array, the gradient of the model at the sample point
        """
        # Evaluate the gradient of the prior
        dy = self.prior.gradient(x)
        dy = np.reshape(dy, (dy.shape[0], -1))
        # Evaluate the kernel gradient
        if self._sample_points is not None:
            dk = self.kernel.gradient(x, self._sample_points)
            w = np.reshape(self._kernel_weights, (self._kernel_weights.shape[0], -1))
            dy = dy + w.T.dot(dk)
        return dy

    @property
    def model_errors(self):
        if self._sample_points is None:
            return None
        K = self.kernel(self._sample_points)
        return K.dot(self._kernel_weights)

    @property
    def num_samples(self):
        if self._sample_points is None:
            return 0
        else:
            return self._sample_points.shape[1]

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

class ContactModel(_ContactModel):
    def __init__(self, surface, friction):
        assert issubclass(type(surface), DifferentiableModel), 'surface must be a subclass of DifferentiableModel'
        assert issubclass(type(friction), DifferentiableModel), 'friction must be a subclass of DifferentiableModel'
        self.surface = surface
        self.friction = friction

    @classmethod
    def FlatSurfaceWithConstantFriction(cls, location = 0., friction = 1.0, direction=np.array([0., 0., 1.])):
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

class SemiparametricContactModel(ContactModel):
    def __init__(self, surface, friction):
        assert isinstance(surface, SemiparametricModel), 'surface must be a semiparametric model'
        assert isinstance(friction, SemiparametricModel), 'friction must be a semiparametric model'
        super(SemiparametricContactModel, self).__init__(surface, friction)

    @classmethod
    def FlatSurfaceWithRBFKernel(cls, height=0., friction=1., length_scale=0.1):
        """
        Factory method for constructing a semiparametric contact model
        
        Assumes the prior is a flat surface with constant friction
        uses independent RBF kernels for the surface and friction models
        """
        surf = SemiparametricModel.FlatPriorWithRBFKernel(location = height, length_scale=length_scale)
        fric = SemiparametricModel.ConstantPriorWithRBFKernel(const = friction, length_scale=length_scale)
        return cls(surf, fric)

    @classmethod
    def FlatSurfaceWithHuberKernel(cls, height = 0., friction = 1., length_scale = 0.1, delta = 0.1):
        """
        Factory method for constructing a semiparametric contact model
        Assumes the prior is a flat surface with constant friction
        Uses independent Pseudo-Huber kernels for the surface and friction models
        """
        surf = SemiparametricModel.FlatPriorWithHuberKernel(location = height, length_scale=length_scale, delta=delta)
        fric = SemiparametricModel.ConstantPriorWithHuberKernel(const = friction, length_scale=length_scale, delta=delta)
    
        return cls(surf, fric)

    def add_samples(self, sample_points, surface_weights, friction_weights):
        """
        Add samples to the semiparametric model

        """
        self.surface.add_samples(sample_points, surface_weights)
        self.friction.add_samples(sample_points, friction_weights)

    @property
    def surface_kernel(self):
        return self.surface.kernel

    @property
    def friction_kernel(self):
        return self.friction.kernel

if __name__ == '__main__':
    print("Hello from contactmodel.py!")