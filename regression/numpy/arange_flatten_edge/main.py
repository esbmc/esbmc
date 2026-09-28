import numpy as np

# flatten()/ravel() on a 1-D arange(...) constructor call used directly as
# the argument (not via a variable) is a degenerate case (already flat), but
# it must still be resolved instead of failing extraction.
a = np.flatten(np.arange(3))
assert a[2] == 2

b = np.ravel(np.arange(3))
assert b[1] == 1
