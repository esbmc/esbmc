import numpy as np

# A fourth positional argument is the dtype, as in NumPy, so 4 is rejected
# as a dtype rather than read as a fourth bound.
a = np.arange(1, 2, 3, 4)
assert len(a) == 0
