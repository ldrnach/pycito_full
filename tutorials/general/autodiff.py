from pydrake.autodiffutils import AutoDiffXd
import numpy as np

# With AutoDiff, the first argument is the value and the second is the current derivative

x_ad = np.array([AutoDiffXd(2.0, np.array([1.0, 0.0])), AutoDiffXd(1.5, np.array([0.0, 1.0]))], dtype=object)
# Function operation
y_ad = (1.0 + x_ad)**2
# Evaluate function and derivative
print(f"Value: {[y_ad[0].value(), y_ad[1].value()]}")
print(f"Derivative: {np.vstack([y_ad[0].derivatives(), y_ad[1].derivatives()])}")

