import numpy as np

w = np.linalg.eigvals(np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]))
assert w[0].imag == 0.0
