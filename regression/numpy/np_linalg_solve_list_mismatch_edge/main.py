import numpy as np

caught = 0
try:
    x = np.linalg.solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0])
except ValueError:
    caught += 1
a = np.array([[1.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, 2.0, 3.0])
try:
    y = np.linalg.solve(a, b)
except ValueError:
    caught += 1
assert caught == 2
