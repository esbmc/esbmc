import numpy as np

values = np.arccos([1.0, 0.999999, -0.999999, -1.0])

assert values[0] < 1e-6
assert values[1] < 0.0015
assert values[2] > 3.13
assert values[3] > 3.13
