import numpy as np

a = np.linspace(0, 1, 4, endpoint=False)
assert len(a) == 4
assert a[1] == 0.25 and a[3] == 0.75
