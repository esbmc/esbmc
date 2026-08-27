import numpy as np

a = np.array([[True, False], [False, False]])
t = np.reshape(a, (4,))

assert t.any()
assert not t.all()
