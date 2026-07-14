import numpy as np
m = np.array([[0 + 1j, 0 + 0j], [0 + 0j, 0 + 1j]])
x = np.linalg.det(m)
assert x.real == -1.0
assert x.imag == 0.0
