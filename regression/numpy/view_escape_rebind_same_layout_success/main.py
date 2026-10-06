import numpy as np

# A view name bound again to a view with the same layout over the same
# array: the pointer moves, the metadata is unchanged.
a = np.array([[1, 2, 3], [4, 5, 6]])
v = a[0]
v = a[1]
assert v[0] == 4
v[1] = 9
assert a[1][1] == 9
assert a[0][1] == 2
