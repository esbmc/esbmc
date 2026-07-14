import numpy as np

values = np.arccos([1.0, 0.0, -1.0])

assert values[0] < 1e-6
assert values[1] > 1.5 and values[1] < 1.6
assert values[2] > 3.13
