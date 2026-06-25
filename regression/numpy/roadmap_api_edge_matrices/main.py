import numpy as np

assert np.full((1, 3), -0.0) == [[-0.0, -0.0, -0.0]]
assert np.eye(3, 2) == [[1, 0], [0, 1], [0, 0]]
