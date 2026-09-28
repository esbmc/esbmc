import numpy as np

# The last-element slice must map to the correct offset within the base
# buffer -- an off-by-one in the pointer/offset computation would either
# write one element short or run past the end of the base array.
a = np.array([10, 20, 30, 40])
v = a[3:4]
v[0] = 99

assert a[3] == 99
assert len(v) == 1
