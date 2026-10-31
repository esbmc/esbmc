import numpy as np

x = nondet_float()
w, v = np.linalg.eig(np.array([[x, 1.0], [1.0, 2.0]]))
