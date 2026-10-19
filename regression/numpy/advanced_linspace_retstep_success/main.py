import numpy as np

a, step = np.linspace(0, 1, 5, retstep=True)
assert step == 0.25
assert a[4] == 1.0 and len(a) == 5
