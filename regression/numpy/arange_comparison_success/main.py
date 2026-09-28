import numpy as np

# np.arange(...) used directly as a comparison argument (not via a variable)
# previously failed extraction the same way the reducer/flatten consumers did.
a = np.greater(np.arange(4), 1)
assert a == [False, False, True, True]

b = np.equal(np.arange(3), [0, 2, 2])
assert b == [True, False, True]
