import numpy as np

x = np.array([0.0, 1.0, 2.0])
np.expm1(x, where=[True, False], out=np.zeros(3))
