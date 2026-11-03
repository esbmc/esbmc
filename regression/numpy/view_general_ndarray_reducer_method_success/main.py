import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert v.sum() == 6
assert v.min() == 1
assert v.max() == 4
assert v.argmin() == 0
assert v.argmax() == 1
assert v.any()
assert v.all()
