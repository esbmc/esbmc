import numpy as np

# A repeated row, or a zero row or column, gives an exact zero pivot.
assert np.linalg.det(np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])) == 0.0
assert np.linalg.det(np.array([[0.0, 0.0], [3.0, 7.0]])) == 0.0
assert np.linalg.det(np.array([[0.0, 5.0], [0.0, 7.0]])) == 0.0
near = np.linalg.det(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
assert abs(near) < 1e-9
