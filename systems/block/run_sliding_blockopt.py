"""
Run a sliding trajectory optimization for the sliding block

Luke Drnach

"""

import numpy as np
from systems.block.blockopt import BlockOptimizer

# Run block optimization
options = BlockOptimizer.defaultOptimizationOptions()
options.useNonlinearComplementarity()
optimizer = BlockOptimizer(num_timepoints=101, options=options)
optimizer.setBoundaryConditions([0., 0.5, 0., 0.], [5., 0.5, 0., 0.])
optimizer.useLinearGuess()
optimizer.enforceEqualTimesteps()
optimizer.setSolverOption("Scale Option", 2)
#optimizer.trajopt.enforceNormalDissipation()
optimizer.setControlWeights(weights=[10.])
optimizer.setStateWeights(weights=[1.0, 1.0, 1.0, 1.0], ref = [5., 0.5, 0., 0.])
optimizer.enableDebugging()
# Solve the problem
optimizer.finalizeProgram()
cost = lambda h: np.sum(h)
optimizer.trajopt.add_final_cost(cost, vars=[optimizer.trajopt.h], name="TotalTime")
result, elapsed = optimizer.solve()
# Print the results
optimizer.plot(result)
optimizer.plotConstraints(result)