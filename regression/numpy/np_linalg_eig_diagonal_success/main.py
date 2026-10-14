import numpy as np

a = np.array([[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
vals = np.linalg.eig(a)

assert vals[0] == 4.0
assert vals[1] == 2.0
assert vals[2] == 1.0
