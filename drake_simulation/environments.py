import numpy as np
from pydrake.all import RigidTransform, CoulombFriction, HalfSpace, Box

class FlatGroundEnvironment():
    """Create a flat ground (halfspace) contact environment with constant friction"""
    def __init__(self, friction = 1.0):
        self._friction = friction

    def addEnvironmentToPlant(self, plant):
        """
        Add a halfspace contact environment with constant friction to the plant model
        """
        T_BG = RigidTransform()
        friction = CoulombFriction(
            static_friction = self.friction,
            dynamic_friction = self.friction
        )
        plant.RegisterCollisionGeometry(
            plant.world_body(),         # Body for which this object is registered
            T_BG,                       # Fixed pose of geometry frame G in body frame B
            HalfSpace(),                # Geometry of the object
            'ground_collision',         # Name
            friction                    # Coulomb friction coefficients
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            HalfSpace(),
            'ground_visual',
            np.array([0.5, 0.5, 0.5, 1.0])  # Color set to opaque
        )
        return plant

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, val):
        assert isinstance(val, (int, float)) and val >= 0, 'friction must be a nonnegative scalar int or float'
        self._friction = val

class FlatGroundWithFrictionPatch(FlatGroundEnvironment):
    """
    Create a flat ground environment with a sudden change in friction
    """
    def __init__(self, friction=1.0, patch_friction=0.1, patch_shape = (1.0, 1.0, 0.01), patch_location = (1.5, 0., 0.005)):
        super().__init__(friction = friction)
        self._patch_friction = patch_friction
        self._patch_shape = patch_shape
        self._patch_location = patch_location

    def addEnvironmentToPlant(self, plant):
        """
        Add:
            1. a halfspace with constant friction
            2. a thin box with different friction 
        to the contact environment
        """
        plant = super().addEnvironmentToPlant(plant)
        T_BG = RigidTransform(p = np.array(self._patch_location))
        box = Box(width = self._patch_shape[0], depth = self._patch_shape[1], height=self._patch_shape[2])
        friction = CoulombFriction(
            static_friction = self._patch_friction,
            dynamic_friction = self._patch_friction
        )
        # Register the patch as new collision & visual geometry
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            T_BG,
            box,
            'friction_patch_collision',
            friction
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            box,
            'friction_patch_visual',
            np.array([0.0, 0.0, 0.5, 0.5])  
        )
        return plant

    @property
    def patch_friction(self):
        return self._patch_friction

    @patch_friction.setter
    def patch_friction(self, val):
        assert isinstance(val, (int, float)) and val >= 0, 'patch friction must be a nonnegative int or float'
        self._patch_friction = val

    @property
    def patch_location(self):
        return self._patch_location

    @patch_location.setter
    def patch_location(self, val):
        assert isinstance(val, tuple) and len(val) == 3, 'patch_location be a 3-tuple of ints or floats'
        for loc in val:
            assert isinstance(loc, (int, float)), 'each element in patch_location must be an int or float'
        self._patch_location = val

    @property
    def patch_shape(self):
        return self._patch_shape

    @patch_shape.setter
    def patch_shape(self, val):
        assert isinstance(val, tuple) and len(val) == 3, 'patch_shape be a 3-tuple of ints or floats'
        for loc in val:
            assert isinstance(loc, (int, float)), 'each element in patch_shape must be an int or float'
        self._patch_width = val
