import numpy as np

w = np.linalg.eigvals(np.array([[4.0, 1.0, 2.0], [0.5, 3.0, 1.0], [1.0, 0.0, 1.0]]))
assert w[0].imag == 0.0 and w[1].imag == 0.0 and w[2].imag == 0.0
assert abs(w[0].real + w[1].real + w[2].real - 8.0) < 1e-9
assert abs(w[0].real * w[1].real * w[2].real - 6.5) < 1e-9
largest = max(w[0].real, w[1].real, w[2].real)
assert abs(largest - 4.907954157) < 1e-6
