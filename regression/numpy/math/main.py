import numpy as np

# numpy.ceil
assert np.ceil(4.2) == 5
ce = np.ceil(10.2)
assert ce == 11
assert np.ceil(-1.3) == -1
assert np.ceil(7.0) == 7

a = np.array([1.2, 2.5, -3.7])
result = np.ceil(a) # This call is handled by ceil_array in libm/ceil.c
assert result[0] == 2.0
assert result[1] == 3.0
assert result[2] == -3.0

assert np.floor(2.9) == 2.0
fl = np.floor(5.1)
assert fl == 5
assert np.floor(-2.1) == -3.0

assert np.fabs(-2.1) == 2.1
assert np.fabs(2.1) == 2.1
assert np.fabs(-3) == 3 
assert np.fabs(3) == 3

# pi = 3.141592653589793
#assert np.sin(1.57079633) == 1.0

assert np.exp(0) == 1.0

#assert np.exp(1) == 2.718281828459045

