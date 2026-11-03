import numpy as np

# The rounding bound overflows, so the result is left unconstrained; NumPy
# gives 9.00000000000015.
a = np.array([[1e80, 0.0, 0.0, 0.0], [0.0, 3e-79, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0],
              [0.0, 0.0, 0.0, 3.0]])
assert np.linalg.det(a) == 9.0
