import numpy as np

a = np.array([[True, True], [True, True]])
r = np.reshape(a, (4,))

assert r.all()
