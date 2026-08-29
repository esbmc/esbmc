import numpy as np

# A 1-D slice view must share the base array's buffer: writing through the
# view must mutate the base array, not a private copy.
a = np.array([1, 2, 3, 4])
v = a[1:3]
v[0] = 99

assert a[1] == 99
assert v[0] == 99
assert a[0] == 1
assert a[2] == 3
