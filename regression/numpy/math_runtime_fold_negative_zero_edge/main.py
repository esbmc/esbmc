import numpy as np

a = np.array([-0.0, 0.0])
b = np.sin(a)
assert np.copysign(1.0, b[0]) == -1.0
assert np.copysign(1.0, b[1]) == 1.0
