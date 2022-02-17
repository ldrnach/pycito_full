"""
contactmodel.py: module for specifying arbitrary contact models for use with TimeSteppingRigidBodyPlant

Luke Drnach
February 16, 2022
"""

#TODO: Unittesting
import numpy as np
from abc import ABC, abstractmethod

class _ContactModel(ABC):
    """
    abstract class outlining methods required for specifying contact model geometry and friction characteristics
    """
    @abstractmethod
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

    @abstractmethod
    def eval_friction(self, pt):
        """
        Returns the value of the level sets of the surface friction at the supplied point. Note that the values of friction returned by eval_friction may only be considered accurate when eval_surface(pt) = 0
        
        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates
        
        Return Values
            out: a (1,) numpy array, the friction coefficient evaluation
        """

    @abstractmethod
    def local_frame(self, pt):
        """
        Return the local coordinate frame of the surface geometry at the specified point. 

        Arguments:
            pt: a (3,) numpy array, specifying a point in world coordinates

        Return values:
            R: a (3,3) numpy array. The first row is the surface normal vector. The next two rows are the surface tangent vectors
        """

class ContactModel(_ContactModel):
    def __init__(self, height, friction):
        self._height_fun = height
        self._friction_fun = friction

    def eval_surface(self, pt):
        return self._height_fun(pt)

    def eval_friction(self, pt):
        return self._friction_fun(pt)
    
    def local_frame(self, pt):
        pass

