import numpy as np

# find_var_decl() returns the first textual assignment to a name, not the
# one that actually reaches this use site. A reassigned name used as a
# chained call's argument must not silently resolve to that stale first
# value; it must raise an explicit diagnostic instead.
a = [1, 2, 3]
b = [1, 5, 3]
x = np.equal(a, b)
x = np.not_equal(a, b)
r = np.logical_not(x)
assert len(r) == 3
