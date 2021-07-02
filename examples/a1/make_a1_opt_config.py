"""
General Script for making an optimization configuration for A1

Luke Drnach
July 2, 2021
"""

from trajopt.optimizer import A1OptimizerConfiguration

config = A1OptimizerConfiguration.defaultStandingConfig()
config.save("examples/a1/runs/standing_config.pkl")