import numpy as np

w = np.linalg.eigvals(
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0],
              [0.0, 0.0, 0.0, 4.0]]))
assert w[0].real == 1.0
assert w[1].real == -2.0
assert w[3].real == 4.0
assert w[3].imag == 0.0
