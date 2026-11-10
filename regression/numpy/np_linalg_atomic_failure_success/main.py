import numpy as np

d = 7.0
try:
    d = np.linalg.det(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
except np.linalg.LinAlgError:
    pass
assert d == 7.0
b = np.array([[9.0, 9.0], [9.0, 9.0]])
try:
    b = np.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
except np.linalg.LinAlgError:
    pass
assert b[0][0] == 9.0 and b[1][1] == 9.0
