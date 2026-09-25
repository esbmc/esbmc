import numpy as np

# Comparing two lists of different lengths must raise an explicit diagnostic
# instead of an out-of-bounds JSON access when the shorter list is indexed
# past its end.
a = [1, 2, 3]
b = [1, 2]
r = np.equal(a, b)
assert len(r) == 3
