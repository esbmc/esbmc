import numpy as np

w = np.linalg.eigvals(np.array([[6.0, 0.0, 0.0], [5.0, 4.0, 0.0], [3.0, 2.0, 1.0]]))
assert w[0].real + w[1].real + w[2].real == 11.0
assert w[0].real * w[1].real * w[2].real == 24.0
assert w[0].real == 1.0 or w[0].real == 4.0 or w[0].real == 6.0
