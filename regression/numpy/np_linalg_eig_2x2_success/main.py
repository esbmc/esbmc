import numpy as np

a = np.array([[2.0, 1.0], [1.0, 2.0]])
vals = np.linalg.eig(a)

assert vals[0] == 3.0
assert vals[1] == 1.0
