import numpy as np

a = np.array([[2.0, 1.0], [1.0, 2.0]])
vals = np.linalg.eigvals(a)

assert abs(vals[0] + vals[1] - 4.0) < 1e-9
assert abs(vals[0] - 3.0) < 1e-9 or abs(vals[0] - 1.0) < 1e-9
assert vals[0].imag == 0.0
