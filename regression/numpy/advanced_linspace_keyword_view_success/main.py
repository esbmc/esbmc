import numpy as np

a = np.linspace(start=0.0, stop=4.0, num=5)
v = a[::2]
a[2] = 9.0
assert v[1] == 9.0 and v[2] == 4.0
