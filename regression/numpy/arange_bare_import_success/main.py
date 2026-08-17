from numpy import arange

# A bare-name call (from numpy import arange) must fast-path materialize the
# same as the np.arange(...) attribute form.
a = arange(3)
assert len(a) == 3
assert a[2] == 2
