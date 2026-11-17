import numpy as np

a = np.array([1, 2])
f = np.zeros(2)
np.add(a, 0.5, out=(f,))
assert f[0] == 1.5 and f[1] == 2.5
g = np.zeros(2)
np.add(a, a, g)
assert g[1] == 4.0
h = np.add(a, a, out=None)
assert h[1] == 4
x = np.array([1.0, 2.0])
np.multiply(x, x, out=x)
assert x[0] == 1.0 and x[1] == 4.0
y = np.array([1.0, 2.0, 3.0])
np.add(y[1:], 10.0, out=y[:2])
assert y[0] == 12.0 and y[1] == 13.0 and y[2] == 3.0
