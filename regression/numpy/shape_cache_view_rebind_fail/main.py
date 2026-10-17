import numpy as np

x = np.eye(4)
y = x[1]
x = y
assert x.size == 16
