import numpy as np

b = np.array([[True, False, True], [False, False, True]])

any0 = np.any(b, axis=0)

assert any0[1] == True
