import numpy as np

vals = [0.0, 1.0]
out = np.sin(vals)
assert out[0] == 0.0
assert out[1] > 0.84
