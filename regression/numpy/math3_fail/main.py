import numpy as np

assert np.fabs(0.0) == 0.0
assert np.fabs(-0.0) == 0.0
assert np.fabs(float('inf')) == float('inf')
assert np.fabs(float('-inf')) == float('inf')


