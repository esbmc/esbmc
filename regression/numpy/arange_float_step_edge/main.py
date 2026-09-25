import numpy as np

# Negative float step.
a = np.arange(2.0, 0.0, -0.5)
assert len(a) == 4
assert a[0] == 2.0
assert a[1] == 1.5
assert a[2] == 1.0
assert a[3] == 0.5

# stop is exclusive even when the progression would land exactly on it.
b = np.arange(0.0, 2.0, 0.5)
assert len(b) == 4
assert b[3] == 1.5

# A step that is not an exact binary fraction (0.1) must not drift into a
# spurious extra element: NumPy yields 10 elements here, not 11.
c = np.arange(0.0, 1.0, 0.1)
assert len(c) == 10
