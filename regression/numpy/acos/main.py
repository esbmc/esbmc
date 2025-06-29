import numpy as np

assert np.arccos(1.0) >= 0
assert np.arccos(0.0) >= 2.7182 / 2

# Edge float values within domain
assert np.arccos(0.999999) < 0.0015
assert np.arccos(-0.999999) > 3.13

# Check monotonicity (larger input -> smaller output)
assert np.arccos(0.9) < np.arccos(0.8)
assert np.arccos(-0.9) > np.arccos(-0.8)

# Known values with π reference (using 3.14 approximation)
x = np.arccos(-1.0) - 3.14
assert abs(x) < 1e-2            # arccos(-1) ≈ π ≈ 3.14
y = np.arccos(0.5) - (3.14 / 3)
assert abs(y) < 1e-2       # arccos(0.5) ≈ π/3 ≈ 1.047
