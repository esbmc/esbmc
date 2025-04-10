import numpy as np

# numpy.ceil
assert np.ceil(4.2) == 5
assert np.ceil(-1.3) == -1
assert np.ceil(7.0) == 7

a = np.array([1.2, 2.5, -3.7])
result = np.ceil(a) # This call is handled by ceil_array at libm/ceil.c
assert result[0] == 2.0
assert result[1] == 3.0
assert result[2] == -3.0
