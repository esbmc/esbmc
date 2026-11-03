import numpy as np

a = np.linspace(start=0, stop=1, num=5)
assert len(a) == 5
assert a[1] == 0.25 and a[4] == 1.0
