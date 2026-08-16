import numpy as np

# np.arange(0) must produce an empty array quickly, and np.arange(1) a
# single-element array -- both boundary sizes for the new fast constant path.
a = np.arange(0)
assert len(a) == 0

b = np.arange(1)
assert len(b) == 1
assert b[0] == 0
