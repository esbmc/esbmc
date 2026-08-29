import numpy as np

# Rebinding the base array to a brand-new object must not change what a
# live view sees: `a = np.array(...)` creates a new array object, it does
# not mutate the old one in place, so v must keep observing the values it
# saw right before the rebind (real NumPy object-identity semantics).
a = np.array([1, 2, 3, 4])
v = a[1:3]
a = np.array([9, 9, 9, 9])

assert v[0] == 2
assert v[1] == 3
assert a[0] == 9
