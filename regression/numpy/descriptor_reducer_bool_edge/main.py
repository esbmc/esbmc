import numpy as np

a = np.array([[True, False], [False, False]])
t = np.transpose(a)

assert t.any()
assert not t.all()
