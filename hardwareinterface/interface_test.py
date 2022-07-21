import numpy as np
import unittest, pickle, sys, os
from a1estimatorinterface import A1ContactEstimationInterface
import pycito.systems.contactmodel as cm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lcmscripts'))

SAMPLEFILE = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__))), 'lcm_slice.pkl')

class EstimationInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create the estimation interface"""
        cls.estimator = A1ContactEstimationInterface()

    def get_lcm_sample(self):
        """"Load and return the sample lcm message"""
        with open(SAMPLEFILE, 'rb') as file:
            lcm = pickle.load(file)
        return lcm

    def test_get_configuration(self):
        """Test that the JSON configuration loads correctly"""
        config = self.estimator._get_configuration()
        keychecks = ['Estimator','Solver','FrictionModel','SurfaceModel']
        for keycheck in keychecks:
            self.assertTrue(keycheck in config, f'configuration has no key {keycheck}')

    def test_generate_initial_state(self):
        """Check that the interface generates an initial state with nonnegative normal distances"""
        # Generate initial state
        config = {'InitialLegPose': [0.0, 0.8, -1.6]}
        initial_state = self.estimator._generate_a1_initial_state(config)
        # Check that normal distances are all nonnegative
        context = self.estimator.a1.multibody.CreateDefaultContext()
        self.estimator.a1.multibody.SetPositionsAndVelocities(context, initial_state)
        dist = self.estimator.a1.GetNormalDistances(context)
        self.assertTrue(np.all(dist >= 0), 'generate_initial_state returns initial state with negative contact distance')

    def test_calculate_ground_slope(self):
        """Check that we can accurately calculate the ground slope"""
        # Test when the ground slope is zero
        model = cm.ContactModel(
            surface = cm.FlatModel(location = 0., direction = np.array([0, 0, 1])),
            friction = cm.ConstantModel(1.)
        )
        slope = self.estimator._calculate_ground_slope(model, 0)
        self.assertAlmostEqual(slope[1], 0., delta=1e-6, msg='Calculated ground slope is not accurate for zero slope')
        # Test when the ground slope is nonzero
        true_slope = -np.pi/6
        model.surface._direction = np.array([np.sin(true_slope), 0, np.cos(true_slope)])
        est_slope = self.estimator._calculate_ground_slope(model)
        self.assertAlmostEqual(est_slope[1], true_slope, delta=1e-6, msg=f'Calculated slope {est_slope:.4f} does not match the true slope {true_slope:.4f}')

    def test_lcm_to_arrays(self):
        """Check that we can convert an LCM message to a set of data arrays"""
        
        mbody = self.estimator.a1.multibody
        nX = mbody.num_positions() + mbody.num_velocities()
        nU = mbody.num_actuators()

        msg = self.get_lcm_sample()

        t, x, u = self.estimator._lcm_to_arrays(msg)
        self.assertEqual(t.shape, (), msg='lcm_to_array returns time array of the wrong shape')
        self.assertEqual(x.shape, (nX,), msg="lcm_to_arrays returns a state array of the wrong shape")
        self.assertEqual(u.shape, (nU,), msg='lcm_to_array returns a control array of the wrong shape')

    def test_estimate(self):
        """Check that we can pass an lcm message to estimate and get back a float"""
        msg = self.get_lcm_sample()
        slope, _ = self.estimator.estimate(msg)
        print(f'Estimated slope: {slope:.4f}')
        self.assertTrue(isinstance(slope, float), 'returned slope is not a float')

if __name__ == '__main__':
    unittest.main()