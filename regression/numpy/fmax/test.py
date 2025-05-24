import numpy as np

# Regular comparisons
assert np.fmax(1.0, 2.0) == 2.0
assert np.fmax(3.0, -1.0) == 3.0

# Symmetry check
assert np.fmax(2.0, 1.0) == 2.0
assert np.fmax(-1.0, -3.0) == -1.0

# Zero comparisons
assert np.fmax(0.0, -0.0) == 0.0
assert np.fmax(-0.0, 0.0) == 0.0


