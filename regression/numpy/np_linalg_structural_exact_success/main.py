import numpy as np

# Exact in every IEEE implementation: power-of-two pivots, exact squares and
# a correctly rounded square root.
b = np.linalg.inv(np.array([[2.0, 0.0], [0.0, 4.0]]))
assert b[0][0] == 0.5 and b[1][1] == 0.25 and b[0][1] == 0.0
assert np.linalg.norm(np.array([3.0, 4.0])) == 5.0
assert np.linalg.norm(np.array([[1.0, -2.0], [3.0, 4.0]]), 1) == 6.0
