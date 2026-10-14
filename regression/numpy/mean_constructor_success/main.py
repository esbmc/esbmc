import numpy as np

# np.mean(a) and a.mean() with a built by a shape constructor.
a = np.identity(2)
b = np.mean(a)
assert b == 0.5
c = a.mean()
assert c == 0.5

# A different constructor with a different concrete value.
d = np.ones((2, 3))
e = np.mean(d)
assert e == 1.0
f = d.mean()
assert f == 1.0
