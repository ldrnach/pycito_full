from utilities import load
from trajopt.optimizer import A1VirtualBaseOptimizer

data = load("examples/a1/runs/Jul-02-2021/run_1/trajoptresults.pkl")
optimizer = A1VirtualBaseOptimizer.buildFromConfig("examples/a1/runs/standing_config.pkl")
optimizer.plotConstraints(data, show=True, savename=None)