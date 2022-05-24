import os 
import numpy as np
import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.trajopt.constraints import SemiImplicitEulerDynamicsConstraint as constraint


SOURCE = os.path.join('examples','a1','simulation_tests','fullstep','timestepping','simdata.pkl')
data = utils.load(SOURCE)

a1 = A1VirtualBase()
a1.terrain.friction = 1.0
a1.Finalize()
context = a1.multibody.CreateDefaultContext()
err  = []
jlforce = np.zeros((a1.num_joint_limits, data['force'].shape[1]))
forces = np.concatenate([data['force'], jlforce], axis=0)

for k, dt in enumerate(np.diff(data['time'])):
    viol = constraint.eval(a1, context, dt, data['state'][:, k], data['state'][:, k+1], data['control'][:, k+1], forces[:, k+1])
    print(f'Maximum violation: {np.max(np.abs(viol)):.2e}')