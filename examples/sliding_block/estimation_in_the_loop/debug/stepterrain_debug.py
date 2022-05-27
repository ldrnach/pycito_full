import os
from pycito.controller.optimization import OptimizationLogger


SOURCE = os.path.join("examples","sliding_block","estimation_in_the_loop","paper_final","stepterrain","campc_logs")
FILES = ['mpclogs.pkl']

for file in FILES:
    logger = OptimizationLogger.load(os.path.join(SOURCE, file))
    print(f"Loaded {file}")
    for k, log in enumerate(logger.logs):
        if not log['success']:
            print(f"Optimization failed at index {k} with exit code {log['exitcode']}")
            print(f"\tCost Functions:")
            for key, value in log['costs'].items():
                print(f'\t\t{key}: {value.item():.2e}')
            print(f"\tConstraint Violations:")
            for key, value in log['constraints'].items():
                print(f'\t\t{key}: {value.item():.2e}')
print('finished')
