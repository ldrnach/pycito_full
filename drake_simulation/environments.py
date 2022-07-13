import numpy as np
from pydrake.all import RigidTransform, CoulombFriction, RollPitchYaw, Box

class FlatGroundEnvironment():
    """Create a flat ground (halfspace) contact environment with constant friction"""
    def __init__(self, friction = 1.0, shape = (20, 20, 0.1)):
        self._friction = friction
        self._shape = shape


    def addEnvironmentToPlant(self, plant):
        """
        Add a halfspace contact environment with constant friction to the plant model
        """
        T_BG = RigidTransform(p = np.array([0, 0, -self._shape[-1]/2]))
        box = Box(width = self.shape[0], depth = self.shape[1], height = self.shape[2])
        friction = CoulombFriction(
            static_friction = self.friction,
            dynamic_friction = self.friction
        )
        plant.RegisterCollisionGeometry(
            plant.world_body(),         # Body for which this object is registered
            T_BG,                       # Fixed pose of geometry frame G in body frame B
            box,                # Geometry of the object
            'ground_collision',         # Name
            friction                    # Coulomb friction coefficients
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            box,
            'ground_visual',
            np.array([153/255, 102/255, 51/255, 0.8])  # Color set to opaque
        )
        return plant

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, val):
        assert isinstance(val, (int, float)) and val >= 0, 'friction must be a nonnegative scalar int or float'
        self._friction = val

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        assert isinstance(val, tuple) and len(val) == 3, 'shape be a 3-tuple of ints or floats'
        for loc in val:
            assert isinstance(loc, (int, float)), 'each element in shape must be an int or float'
        self._shape = val

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
        self._patch_shape = val

class RampUpEnvironment(FlatGroundEnvironment):
    def __init__(self, friction=1.0, slope=15, length=1):
        super().__init__(friction)
        self._slope = slope
        self._length = length

    def addEnvironmentToPlant(self, plant):
        """Create the environment collision objects and add them to the plant model"""
        plant = super().addEnvironmentToPlant(plant)
        friction = CoulombFriction(
            static_friction = self.friction,
            dynamic_friction = self.friction,
        )
        # Create and add the ramp
        ramp, T_WR = self.create_ramp()
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            T_WR,
            ramp,
            'ramp_collision',
            friction
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_WR,
            ramp,
            'ramp_visual',
            np.array([102/255, 153/255, 51/255, 0.8])  # Color set to opaque 
        )

        # Create and add the platform
        platform, T_WP = self.create_platform()
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            T_WP,
            platform,
            'platform_collision',
            friction
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_WP,
            platform,
            'platform_visual',
            np.array([102/255, 153/255, 51/255, 0.8])  # Color set to opaque 
        )
        return plant

    def create_ramp(self):
        """Create a box for the ramp and define it's pose relative to the world frame"""
        slope_rad = np.deg2rad(self._slope)
        ramp_height = self.length * np.tan(slope_rad)
        ramp_length = self.length / np.cos(slope_rad)
        ramp = Box(width = ramp_length, depth = 2, height = 0.05)
        pose = RigidTransform(
            p = np.array([1 + self.length/2, 0, ramp_height / 2 - 0.025]),
            rpy = RollPitchYaw(0, -slope_rad, 0)
        )
        return ramp, pose


    def create_platform(self):
        """Create a box for the platform and define it's pose relative to the world frame"""
        platform_height = self.length * np.tan(np.deg2rad(self._slope))
        platform = Box(width = 6, depth = 2, height = 0.05)
        pose = RigidTransform(
            p = np.array([1 + self.length + 3 - 0.025 * np.sin(np.deg2rad(self._slope)), 0, platform_height - 0.025]),
            rpy = RollPitchYaw(0, 0, 0)
        )
        return platform, pose

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, val):
        assert isinstance(val, (int, float)) and val < 90 and val > -90, 'slope must be a scalar integer or float in the interval (-90, 90)'
        self._slope = val

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, val):
        assert isinstance(val, (int, float)) and val > 0, 'length must be a positive int or float'
        self._length = val
    