import numpy as np

a = np.linspace(0, 1, 0)
b, step = np.linspace(0, 1, 0, retstep=True)
assert len(a) == 0 and len(b) == 0
assert step != step
