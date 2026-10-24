import math

import numpy as np

# NumPy takes the remainder from fmod, not from x - floor(x / y) * y.
r = np.array([0.7, -0.7, 0.7, 6.0]) % np.array([0.1, 0.1, -0.1, -3.0])
assert r[0] == 0.09999999999999992
assert r[1] == 8.326672684688674e-17
assert r[2] == -8.326672684688674e-17
assert math.copysign(1.0, r[3]) == -1.0
