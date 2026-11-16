import numpy as np

# Both singular values are sqrt(13); their computed gap rounds to -4.4e-16.
u, s, vh = np.linalg.svd(np.array([[3.0, 2.0], [2.0, -3.0]]))
assert s[0] < 0.0
