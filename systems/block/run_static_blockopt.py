"""
Run a static (no motion) trajectory optimization for the sliding block

Luke Drnach
"""

import numpy as np
from systems.block.blockopt import BlockOptimizer

# Run block optimization
options = BlockOptimizer.defaultOptimizationOptions()
options.useLinearComplementarity()
optimizer = BlockOptimizer(num_timepoints=21, options=options)
optimizer.setBoundaryConditions([0., 0.5, 0., 0.], [0., 0.5, 0., 0.])
optimizer.useLinearGuess()
optimizer.enforceEqualTimesteps()
optimizer.setControlWeights([10.])
optimizer.enableDebugging()
# Solve the problem
optimizer.finalizeProgram()
result, elapsed = optimizer.solve()
# Print the results
optimizer.plot(result)