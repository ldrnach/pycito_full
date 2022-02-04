import pydrake.autodiffutils as ad
import numpy as np

# With AutoDiff, the first argument is the value and the second is the current derivative

x_ad = np.array([ad.AutoDiffXd(2.0, np.array([1.0, 0.0])), ad.AutoDiffXd(1.5, np.array([0.0, 1.0]))], dtype=object)
# Function operation
y_ad = (1.0 + x_ad)**2
# Evaluate function and derivative
print(f"Value: {[y_ad[0].value(), y_ad[1].value()]}")
print(f"Derivative: {np.vstack([y_ad[0].derivatives(), y_ad[1].derivatives()])}")
print(f"Norm: {np.linalg.norm(y_ad)}")

# Checking InitializeAutoDiff functionality
z = np.array([1, 2])
z_ad = ad.InitializeAutoDiff(z)
print(f"InitializeAutoDiff sets value to z = {ad.ExtractValue(z_ad)}")
print(f"InitializeAutoDiff sets gradient to dz = {ad.ExtractGradient(z_ad)}")
