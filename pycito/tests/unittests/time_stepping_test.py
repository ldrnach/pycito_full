"""
test_time_stepping.py: unittests for checking the implementation in TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
import unittest
from pycito.systems.block.block import Block
from pydrake.all import RigidTransform

class TestTimeStepping(unittest.TestCase):
    """ 
    TestTimeStepping 
    
    unit tests for TimeSteppingMultibodyPlant in timestepping.py
    uses a sliding block as a model plant. Checks several methods for consistency with one contact point
    """

    @classmethod
    def setUpClass(cls):
        """Initialize the class with a finalized Block plant"""
        cls._model = Block()
        cls._model.Finalize()  

    def setUp(self):
        """For each unit test, set the position in the plant model"""
        self.context = self._model.multibody.CreateDefaultContext()
        self._model.multibody.SetPositions(self.context, [2.,1.])

    def test_finalized(self):
        """
        Assert that the MultibodyPlant has been finalized
        
        Expected behavior: Drake's MultibodyPlant returns TRUE when finalized, 
                        The TimeSteppingPlant has non-empty collision_poses
        """
        self.assertTrue(self._model.multibody.is_finalized(), msg='MultibodyPlant not finalized')
        self.assertIsNotNone(self._model.collision_poses, msg='Finalize failed to set collision geometries')

    def test_context(self):
        """
        Test that MultibodyPlant can still create contexts
        
        Expected behavior: CreateDefaultContext returns a non-None object
        """
        # Create the context
        context = self._model.multibody.CreateDefaultContext()
        self.assertIsNotNone(context, msg="Context is None")

    def test_set_positions(self):
        """Test that we can still set positions in MultibodyPlant
        
        Expected behavior: The position of the block is [2, 1]
        """
        q = [2.,1.]
        # Now get the position and check it
        self.assertListEqual(self._model.multibody.GetPositions(self.context).tolist(), q, msg="Position not set")

    def test_normal_distances(self):
        """
        Test that the normal distances can be calculated
        
        Expected behavior: At the default position of [2,1], the distance to the terrain is 0.5. GetNormalDistances should return a 1D array with one entry: 0.5
        """
        # Check the values of the distances
        distances = self._model.GetNormalDistances(self.context)
        # Check the values of the distances
        true_dist = np.array([0.5])
        np.testing.assert_allclose(distances, true_dist, err_msg="Incorrect values for normal distances")

    def test_contact_jacobians(self):
        """
        Test the contact jacobians can be calculated
        
        Expected behavior: GetContactJacobians should return 2 2D arrays.
        The first is the normal Jacobian, and should be [0 1] for all positions
        The second is the tangent Jacobian and should have 4 rows. The first two rows are the +X and +Y tangent directions, and should be [1 0] and [0 0] respectively. The remaining rows are the negative of the original rows (the -X and -Y tangent directions)
        """
        # Assert that there is 1 normal jacobian, 4 tangent jacobians
        Jn, Jt = self._model.GetContactJacobians(self.context)
        # Assert the values of the contact Jacobians
        Jn_true = np.array([[0., 1.]])
        Jt_true = np.array([[1., 0.], [0., 0.], [-1., 0.], [0., 0.]])
        np.testing.assert_allclose(Jn, Jn_true, err_msg="Normal Jacobian is incorrect")
        np.testing.assert_allclose(Jt, Jt_true, err_msg="Tangent Jacobian is incorrect")

    def test_friction_coefficients(self):
        """
        Test that the friction coefficients can be calculated
        
        Expected behavior: GetFrictionCoefficients returns the list [0.5], which is the coefficient of friction for the contact point.
        """
        # Get friction coefficients
        friction_coeff = self._model.GetFrictionCoefficients(self.context)
        # Check that the friction coefficient is correct
        self.assertListEqual(friction_coeff, [0.5], msg="wrong number of friction coefficients")

    def test_autodiff_finalized(self):
        """
        Test that the autodiff copy is finalized
        
        Expected behavior: When an autodiff copy is made, it should auto-finalize the underlying multibody plant. The expected behavior is that multibody.is_finalized() returns True.
        """
        # Get the autodiff copy
        copy_ad = self._model.toAutoDiffXd()
        # Check that the copy is finalized
        self.assertTrue(copy_ad.multibody.is_finalized(), msg="AutoDiff copy not finalized")

    def test_autodiff_collisions(self):
        """
        Test that the autodiff copy gets the collision frames
        
        Expected behavior: The autodiff copy should have as many collision frames and poses as the original, float copy.
        """
        # Get the autodiff copy
        copy_ad = self._model.toAutoDiffXd()
        # Check that the number of collision frames and poses is equal to the original
        self.assertEqual(len(copy_ad.collision_frames), len(self._model.collision_frames), msg = "wrong number of collision frames")
        self.assertEqual(len(copy_ad.collision_poses), len(self._model.collision_poses), msg = "wrong number of collision poses")

    def test_duplicator_matrix(self):
        """ 
        Check the value of the duplication matrix
        
        Expected behavior: for a single contact point, the duplication matrix should be a single column of 1s. In this case, D = np.array([[1, 1, 1, 1]])
        """
        D = self._model.duplicator_matrix()
        D_true = np.ones((1,4))
        np.testing.assert_allclose(D, D_true, err_msg="Duplicator matrix is not correct")

    def test_contact_point(self):
        """ 
        Check that the contact point is calculated correctly.

        Expected behavior: at the default position [2, 1], the contact point should be located (in 3D world coordinates) at [2., 0., 0.5]
        """
        contact_pt = self._model.get_contact_points(self.context)
        pt_true = np.array([[2.], [0.], [0.5]])
        np.testing.assert_allclose(contact_pt[0], pt_true, err_msg="get contact points returns incorrect values")

    def test_friction_discretization(self):
        """
            Check the friction disretization matrix for this model

            Expected behavior: At the default discretization level (1), the matrix should be [[1 0 -1 0], [0 1 0 -1]], discretizing X and Y into positive and negative components
        """
        discr = self._model.friction_discretization_matrix()
        discr_true = np.array([[1., 0., -1., 0.],[0., 1., 0., -1.]])
        np.testing.assert_allclose(discr, discr_true, err_msg="friction discretization matrix is incorrect")

    def test_resolve_forces(self):
        """
            Check that we can resolve the positive-only friction forces into forces with potentially negative components.

            Expected behavior: resolve_forces should return the forces in the format (Normal, Tangent1, Tangent2), where Tangent1 = (Tangent_X+ - Tangent_X-) and Tangent2 = (Tangent_Y+ - Tangent_Y-). In this example, that corresponds to [1, 2 - 4, 6 - 3] = [1, -2, 3]
        """

        force = np.array([[1., 2., 6., 4., 3.]]).transpose()
        rforce = self._model.resolve_forces(force)
        force_true = np.array([[1., -2., 3.]]).transpose()
        np.testing.assert_allclose(rforce, force_true, err_msg="Resolved forces are not correct")

    def test_resolve_forces_in_world(self):
        """
        Tests that non-negative contact force variables can be converted into a force vector in world-coordinates, that is as [Fx, Fy, Fz].

        Expected behavior: The expected behavior is that the force is resolved and then converted into [Tangent_X, Tangent_Y, Normal] format. In this example, that corresponds to [2-4, 6-3, 1] = [-2, 3, 1]
        """

        force = np.array([[1., 2., 6., 4., 3.]]).transpose()
        rforce = self._model.resolve_contact_forces_in_world(self.context, force)
        force_true = np.array([[-2., 3., 1.]]).transpose()
        np.testing.assert_allclose(rforce, force_true, err_msg="resolved force in world is not correct")

    def test_resolve_forces_at_point(self):
        """
        Test that we can resolve forces in world coordinates at arbitrary points.
        For this example, the functionality is the same as "test_resolve_forces_in_world" because the ground model is flat.
        """
        force = np.array([[1., 2., 6., 4., 3.]]).transpose()
        pt = np.array([2., 0., 1.])
        rforce = self._model.resolve_contact_forces_in_world_at_points(force, pt)
        force_true = np.array([[-2., 3., 1.]]).transpose()
        np.testing.assert_allclose(rforce, force_true, err_msg="Resolved force at point is incorrect")

class MultiContactTest(unittest.TestCase):
    """
        A second unit test suite for TimeStepping. 
        In this case, we introduce several contact points by making multiple block models.
    """
    @classmethod
    def setUpClass(cls) -> None:
        """ Set up the class by creating a multibody plant model with three blocks """
        cls.model = Block()
        # Add a second block to the model
        cls.model.add_model(urdf_file="systems/block/urdf/sliding_block.urdf", name="box2")
        ind = cls.model.multibody.GetBodyIndices(cls.model.model_index[1])[0]
        body_frame = cls.model.multibody.get_body(ind).body_frame()
        cls.model.multibody.WeldFrames(cls.model.multibody.world_frame(), body_frame, RigidTransform())
        # Add a third block to the model
        cls.model.add_model(urdf_file="systems/block/urdf/sliding_block.urdf", name="box3")
        ind = cls.model.multibody.GetBodyIndices(cls.model.model_index[2])[0]
        body_frame = cls.model.multibody.get_body(ind).body_frame()
        cls.model.multibody.WeldFrames(cls.model.multibody.world_frame(), body_frame, RigidTransform())
        # Finalize the model
        cls.model.Finalize()

    def setUp(self):
        """ 
        Set the position variables for each test
        
        with multiple bodies, the position vector is [x1, x2, x3, z1, z2, z3].

        In this case, we set the positions of the blocks to [1, 1], [-2, 2], and [4, -1] respectively.
        """
        self.context = self.model.multibody.CreateDefaultContext()
        self.model.multibody.SetPositions(self.context, [1., -2., 4., 1., 2., -1.])

    def test_set_positions(self):
        """
        Check that setting the positions worked 
        
        Expected behavior: we get a vector of positions, [1, -2, 4, 1, 2, -1]
        """

        pos = self.model.multibody.GetPositions(self.context)
        pos_true = np.array([1., -2., 4., 1., 2., -1.])
        np.testing.assert_allclose(pos, pos_true, err_msg="Setting positions failed")

    def test_contact_jacobians(self):
        """ 
        Check that the contact jacobians are correct
        
        Expected behavior: GetContactJacobians returns a 3x6 matrix for the normal jacobian and a 12x6 matrix for the tangent jacobian. Both matrices are sparse, with +-1 in places corresponding to Z directions (normal jacobian) and +-X directions (tangent jacobians). Moreover, the rows corresponding to each contact point should be independent, although this is verified indirectly. 
         """
        # Calculate the normal and tangent jacobians
        Jn_true = np.zeros((3, 6))
        for n in range(3):
            Jn_true[n, n+3] = 1.
        Jt_true = np.zeros((12, 6))
        for n in range(3):
            Jt_true[4*n, n] = 1.
            Jt_true[4*n+2,n] = -1.
        # Get the values from the model
        Jn, Jt = self.model.GetContactJacobians(self.context)
        np.testing.assert_allclose(Jn, Jn_true, err_msg="Normal Jacobian incorrect")
        np.testing.assert_allclose(Jt, Jt_true, err_msg="Tangent Jacobian incorrect")

    def test_normal_distances(self):
        """
        Check that the normal distances are correct
        
        Expected behavior: GetNormalDistances should return an array with 3 elements, each corresponding to a single block. Each element should be equal to z_i - 0.5, where z_i is the ith block's height coordinate. In this case, the return should be [1., 2., -1.] - 0.5 = [0.5, 1.5, -1.5]
        """
        distances = self.model.GetNormalDistances(self.context)
        true_dist = np.array([0.5, 1.5, -1.5])
        np.testing.assert_allclose(distances, true_dist, err_msg="Normal distances are incorrect")

    def test_duplicator_matrix(self):
        """ 
        Test the duplicator is correct 
        
        Unlike the single contact point case, the duplicator matrix in this case is 3x12, with independent rows to indicate the independent contact points.
        """
        D = self.model.duplicator_matrix()
        D_true = np.array([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]])
        np.testing.assert_allclose(D, D_true, err_msg="discretization matrix incorrect")

    def test_friction_discretization(self):
        """ Test the friction discretization is correct """
        D = self.model.friction_discretization_matrix()
        D_true = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.]])
        np.testing.assert_allclose(D, D_true, err_msg="friction discretization matrix incorrect")

    def test_resolve_forces(self):
        """ Test forces are resolved into (Normal, Tangential)"""
        test_force = np.array([[1., 2., 3., 4., 5., 6., -9., 10., 11., 10., 9., 13., 16., 14., 12.]]).transpose()
        rforce = self.model.resolve_forces(test_force)
        expected = np.array([[1., 2., 3., -2., 14., 0., 2., -1., 4.]]).transpose()
        np.testing.assert_allclose(rforce, expected, err_msg = "Resolved force incorrect")

    def test_resolve_forces_in_world(self):
        """ test resolving forces in world coordinates"""
        test_force = np.array([[1., 2., 3., 4., 5., 6., -9., 10., 11., 10., 9., 13., 16., 14., 12.]]).transpose()
        rforce = self.model.resolve_contact_forces_in_world(self.context, test_force)
        expected = np.array([[-2., 14., 1.],[0., 2., 2.],[-1., 4., 3.]]).transpose()
        np.testing.assert_allclose(rforce, expected, err_msg="world force incorrect")

    def test_resolve_forces_at_points(self):
        pass

if __name__ == "__main__":
    unittest.main()