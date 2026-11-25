import numpy as np

b = np.eye(2)
try:
    b = np.linalg.inv([[1.0, 2.0, 3.0]])
except np.linalg.LinAlgError:
    pass
assert b[0][0] == 1.0 and b[0][1] == 0.0
