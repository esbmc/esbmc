import numpy as np

# A constructor call carrying a keyword argument (dtype=) is outside the
# conservative materialization this block adds; must keep the existing
# explicit diagnostic instead of silently misreading the shape as data.
a = np.zeros((2, 2), dtype=int)
b = a.flatten()

assert b[0] == 0
