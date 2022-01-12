"""
terrain.py: package for specifying arbitrary terrain geometries for use with TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
from abc import ABC, abstractmethod 
import pycito.systems.gaussianprocess as gp
from pydrake.autodiffutils import initializeAutoDiff, AutoDiffXd

#TODO: Check output of local_frame. Make sure the frame matrix is normalized and the vectors are COLS and not ROWS

class Terrain(ABC):
    """
    Abstract class outlining methods required for specifying a terrain geometry that can be used with TimeSteppingMultibodyPlant
    """
    @abstractmethod
    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: the point of interest
        Return values:
            y: the point on the terrain which is nearest to x
        """

    @abstractmethod 
    def local_frame(self, x):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: a point on the terrain
        Return values:
            R: an array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """

    @abstractmethod
    def get_friction(self, x):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """

    def str(self):
        return f"{type(self).__name__}"

class FlatTerrain2D(Terrain):
    """
    Implementation of a 2-dimensional terrain with flat geometry
    """
    def __init__(self, height = 0, friction = 0.5):
        """ Construct the terrain, set it's height and friction coefficient"""
        self.height = height
        self.friction  = friction

    def str(self):
        return super(FlatTerrain2D, self).str() + f"with height {self.height} and friction {self.friction}"

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (2x1), the point of interest
        Return values:
            y: (2x1), the point on the terrain which is nearest to x
        """
        return np.array([x[0], self.height])

    def local_frame(self, _):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (2x1) a point on the terrain
        Return values:
            R: (2x2), an array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        return np.array([[0,1],[1,0]])
    
    def get_friction(self, _):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: (2x1) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        return self.friction

class FlatTerrain(Terrain):
    """ Implementation of 3-dimensional flat terrain with no slope """
    def __init__(self, height=0.0, friction=0.5):
        """ Constructs the terrain with the specified height and friction coefficient """
        self.height = height
        self.friction = friction
    
    def str(self):
        return super(FlatTerrain, self).str() + f"with height {self.height} and friction {self.friction}"

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        terrain_pt[-1] = self.height
        return terrain_pt

    def local_frame(self, _):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (3x1), a point on the terrain
        Return values:
            R: a (3x3) array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        return np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])

    def get_friction(self, _):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: (3x1) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        return self.friction

class StepTerrain(FlatTerrain):
    """ Implementation of 3-dimensional terrain with a step in the 1st dimension"""
    def __init__(self, height = 0.0, step_height=1.0, step_location=1.0, friction=0.5):
        """ Construct the terrain and set the step x-location and step height"""
        super().__init__(height, friction)
        self.step_height = step_height
        self.step_location = step_location

    def str(self):
        return super(StepTerrain, self).str() + f"with height {self.height}, step height {self.step_height} and location {self.step_location} and friction {self.friction}"

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        if x[0] < self.step_location:
            terrain_pt[-1] = self.height
        else:
            terrain_pt[-1] = self.step_height 
        return terrain_pt           

class SlopeStepTerrain(FlatTerrain):
    """ Implementation of a piecewise linear terrain with a slope"""
    def __init__(self, height, slope, slope_location, friction):
        """ Construct the terrain with the specified slope, starting at x = slope_location """
        super().__init__(height, friction)
        self.slope = slope
        self.slope_location = slope_location

    def str(self):
        return super(SlopeStepTerrain, self).str() + f"with height {self.height} and slope {self.slope} starting at {self.slope_location} and friction {self.friction}"

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        if self._on_flat(x):
            terrain_pt[-1] = self.height
        else:
            terrain_pt[-1] = self.slope * (terrain_pt[0] - self.slope_location) + self.height
        return terrain_pt

    def local_frame(self, x):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (3x1), a point on the terrain
        Return values:
            R: a (3x3) array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        if self._on_flat(x):
            return np.array([[0., 0., 1.],[1., 0., 0.], [0., 1., 0.]])
        else:
            B = np.array([[-self.slope, 0., 1.],[1., 0., self.slope],[0., 1., 0.]])
            return B/np.sqrt(1 + self.slope **2)

    def _on_flat(self, x):
        """ Check if the point is on the flat part of the terrain or not """
        return x[-1] < self.height - self.slope *(x[0] - self.slope_location)

class VariableFrictionFlatTerrain(FlatTerrain):
    def __init__(self, height=0.0, fric_func=None):
        """ Constructs a flat terrain with variable friction """
        super().__init__(height, friction=None)
        if fric_func is None:
            self.friction = ConstantFunc(0.5)
        else:
            self.friction = fric_func

    def str(self):
        return super(VariableFrictionFlatTerrain, self).str() + f"with height {self.height} and variable friction defined by {self.fric_func.__name__}"

    def get_friction(self, x):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: (3x1) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        return self.friction(x)

class GaussianProcessTerrain(FlatTerrain):
    def __init__(self, height_gp=None, friction_gp=None):
        """ Construct a terrain model based on the Gaussian process"""
        super().__init__(height=None, friction=None)
        # Set the default values
        if height_gp is None:
            height_gp = gp.GaussianProcess(xdim=2,
                            mean=ConstantFunc(0.0), 
                            kernel=gp.SquaredExpKernel(M=np.eye(2), s=1.))
        if friction_gp is None:
            friction_gp = gp.GaussianProcess(xdim=2,
                            mean=ConstantFunc(0.5),
                            kernel=gp.SquaredExpKernel(M=np.eye(2), s=1.))
        # Set up the terrain heightmap GP
        self.height = height_gp
        self.friction = friction_gp

    def str(self):
        return super(GaussianProcessTerrain, self).str() + f"with height defined by {str(self.height)} and friction defined by {str(self.friction)}"

    def nearest_point(self, x):
        """
        Returns a point on the expected posterior terrain close to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        if np.ndim(terrain_pt) == 1:
            terrain_pt = np.expand_dims(terrain_pt, axis=1)
        terrain_pt[2], _ = self.height.posterior(terrain_pt[0:2])
        return np.squeeze(terrain_pt)

    def local_frame(self, x):
        """
        Returns the expected posterior local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (3,), a point on the terrain
        Return values:
            R: a (3,3) array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        x = np.expand_dims(x,axis=1)
        dg = self.posterior_gradient(x[0:2,:])
        n = np.array([-dg[0], -dg[1], 1])
        t1 = np.array([1, 0, dg[0]])
        t2 = np.array([0, 1, dg[1]])
        n = n / np.linalg.norm(n, ord=2)
        t1 = t1 / np.linalg.norm(t1, ord=2)
        t2 = t2 / np.linalg.norm(t2, ord=2)
        return np.vstack((n, t1, t2))
        
    def posterior_gradient(self, x):
        x_ad = initializeAutoDiff(x)
        y_ad, _ = self.height.posterior(x_ad)
        return y_ad[0,0].derivatives()

    def get_friction(self, x):
        """
        Returns the expected posterior value of terrain friction coefficient at the supplied point

        Arguments:
            x: (3,) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        x = np.expand_dims(x, axis=1)
        mu, _ = self.friction.posterior(x[0:2,:])
        return mu.item()

def ConstantFunc():
    """Parameterized constant function with AutoDiff type support"""
    def __init__(self, const):
        self.const = const
    def __call__(self,x):
        if isinstance(x[0], AutoDiffXd):
            return AutoDiffXd(self.const, 0.*x[0].derivatives())
        else:
            return self.const

    def str(self):
        return f"{type(self).__name__}"

if __name__ == "__main__":
    print("Hello from terrain.py")