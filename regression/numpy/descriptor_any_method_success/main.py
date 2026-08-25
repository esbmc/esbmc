import numpy as np

a = np.array([[False, False], [True, False]])
t = np.transpose(a)

assert t.any()
