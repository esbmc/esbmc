import numpy as np

caught = False
try:
    b = np.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
except np.linalg.LinAlgError:
    caught = True
assert caught
