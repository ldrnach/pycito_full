import os
import numpy as np

from a1_simulator import A1DrakeSimulationBuilder
from pycito.utilities import load
from pycito.systems.visualization import Visualizer
import environments as env

from pydrake.all import PiecewisePolynomial

SOURCE = os.path.join('drake_simulation','mpc_walking_sim','symmetric_2','simdata.pkl')
URDF = os.path.join('pycito', 'systems','A1','A1_description','urdf','a1_no_collision.urdf')


data = load(SOURCE)

xtraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])

viz = Visualizer(URDF)
viz.plant = env.FlatGroundEnvironment().addEnvironmentToPlant(viz.plant)
viz.visualize_trajectory(xtraj)



