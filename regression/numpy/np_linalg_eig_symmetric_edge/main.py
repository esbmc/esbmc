import numpy as np

a = np.array([[2.0, 0.0], [0.0, 2.0]])
vals = np.linalg.eig(a)

assert vals[0] == 2.0
assert vals[1] == 2.0
