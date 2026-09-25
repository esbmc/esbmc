import numpy as np

# The constructor call used directly as a reducer's argument (not via a
# variable) previously failed extraction: only a Name argument was resolved
# to its declaration, leaving an inline np.arange(...) call untouched.
assert np.sum(np.arange(4)) == 6
assert np.mean(np.arange(4)) == 1.5
assert np.max(np.arange(4)) == 3
