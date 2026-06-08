import numpy as np

vals = [[0.0, 1.0], [4.0, 9.0]]
out = np.sqrt(vals)
assert out[0][0] == 0.0
assert out[1][1] == 3.0
