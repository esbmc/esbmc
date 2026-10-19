import numpy as np

a = np.linspace(0, 0.3, 3, endpoint=False)
b = np.linspace(0, 0.3, 4)
assert len(a) == 3 and a[2] < 0.3
assert b[3] == 0.3
