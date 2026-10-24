import numpy as np

a = np.linspace(2, 5, num=1)
b = np.linspace(2, 5, 1, endpoint=False)
assert len(a) == 1 and a[0] == 2.0
assert b[0] == 2.0
