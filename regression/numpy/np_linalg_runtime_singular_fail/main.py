import numpy as np

x = nondet_float()
if x >= -8.0 and x <= 8.0:
    b = np.linalg.inv(np.array([[x, 2.0 * x], [1.0, 2.0]]))
