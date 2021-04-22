"""
test_time_stepping.py: unittests for checking the implementation in TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
import unittest
from systems.block.block import Block
from pydrake.all import RigidTransform
#TODO: Write tests for testing autodiff functions

class TestTimeStepping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model = Block()
        cls._model.Finalize()  

    def setUp(self):
        """Set the context variable for each test"""
        self.context = self._model.multibody.CreateDefaultContext()
        self._model.multibody.SetPositions(self.context, [2.,1.])

    def test_finalized(self):
        """Assert that the MultibodyPlant has been finalized"""
        self.assertTrue(self._model.multibody.is_finalized(), msg='MultibodyPlant not finalized')
        self.assertIsNotNone(self._model.collision_poses, msg='Finalize failed to set collision geometries')

    def test_context(self):
        """Test that MultibodyPlant can still create Contexts"""
        # Create the context
        context = self._model.multibody.CreateDefaultContext()
        self.assertIsNotNone(context, msg="Context is None")

    def test_set_positions(self):
        """Test that we can still set positions in MultibodyPlant"""
        q = [2.,1.]
        # Now get the position and check it
        self.assertListEqual(self._model.multibody.GetPositions(self.context).tolist(), q, msg="Position not set")

    def test_normal_distances(self):
        """Test that the normal distances can be calculated"""
        # Check the values of the distances
        distances = self._model.GetNormalDistances(self.context)
        # Check the values of the distances
        true_dist = np.array([0.5])
        np.testing.assert_allclose(distances, true_dist, err_msg="Incorrect values for normal distances")

    def test_contact_jacobians(self):
        """Test the contact jacobians can be calculated"""
        # Assert that there is 1 normal jacobian, 4 tangent jacobians
        Jn, Jt = self._model.GetContactJacobians(self.context)
        # Assert the values of the contact Jacobians
        Jn_true = np.array([[0., 1.]])
        Jt_true = np.array([[1., 0.], [0., 0.], [-1., 0.], [0., 0.]])
        np.testing.assert_allclose(Jn, Jn_true, err_msg="Normal Jacobian is incorrect")
        np.testing.assert_allclose(Jt, Jt_true, err_msg="Tangent Jacobian is incorrect")

    def test_friction_coefficients(self):
        """Test that the friction coefficients can be calculated"""
        # Get friction coefficients
        friction_coeff = self._model.GetFrictionCoefficients(self.context)
        # Check that the friction coefficient is correct
        self.assertListEqual(friction_coeff, [0.5], msg="wrong number of friction coefficients")

    def test_autodiff_finalized(self):
        """Test that the autodiff copy is finalized"""
        # Get the autodiff copy
        copy_ad = self._model.toAutoDiffXd()
        # Check that the copy is finalized
        self.assertTrue(copy_ad.multibody.is_finalized(), msg="AutoDiff copy not finalized")

    def test_autodiff_collisions(self):
        """Test that the autodiff copy gets the collision frames"""
        # Get the autodiff copy
        copy_ad = self._model.toAutoDiffXd()
        # Check that the number of collision frames and poses is equal to the original
        self.assertEqual(len(copy_ad.collision_frames), len(self._model.collision_frames), msg = "wrong number of collision frames")
        self.assertEqual(len(copy_ad.collision_poses), len(self._model.collision_poses), msg = "wrong number of collision poses")

    def test_duplicator_matrix(self):
        """ Check the value of the duplication matrix """
        D = self._model.duplicator_matrix()
        D_true = np.ones((1,4))
        np.testing.assert_allclose(D, D_true, err_msg="Duplicator matrix is not correct")

    def test_contact_point(self):
        contact_pt = self._model.get_contact_points(self.context)
        pt_true = np.array([[2.], [0.], [0.5]])
        np.testing.assert_allclose(contact_pt[0], pt_true, err_msg="get contact points returns incorrect values")

    def test_friction_discretization(self):
        discr = self._model.friction_discretization_matrix()
        discr_true = np.array([[1., 0., -1., 0.],[0., 1., 0., -1.]])
        np.testing.assert_allclose(discr, discr_true, err_msg="friction discretization matrix is incorrect")

    def test_resolve_forces(self):
        force = np.array([[1., 2., 6., 4., 3.]]).transpose()
        rforce = self._model.resolve_forces(force)
        force_true = np.array([[1., -2., 3.]]).transpose()
        np.testing.assert_allclose(rforce, force_true, err_msg="Resolved forces are not correct")

    def test_resolve_forces_in_world(self):
        force = np.array([[1., 2., 6., 4., 3.]]).transpose()
        rforce = self._model.resolve_contact_forces_in_world(self.context, force)
        force_true = np.array([[-2., 3., 1.]]).transpose()
        np.testing.assert_allclose(rforce, force_true, err_msg="resolved force in world is not correct")

    def test_resolve_forces_at_point(self):
        force = np.array([[1., 2., 6., 4., 3.]]).transpose()
        pt = np.array([2., 1.])
        rforce = self._model.resolve_contact_forces_in_world_at_points(force, pt)
        force_true = np.array([[-2., 3., 1.]]).transpose()
        np.testing.assert_allclose(rforce, force_true, err_msg="Resolved force at point is incorrect")



if __name__ == "__main__":
    unittest.main()