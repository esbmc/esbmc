import numpy as np

# np.sum(a) and a.sum() with a built by a shape constructor (not a
# np.array(...) literal).
a = np.identity(3)
b = np.sum(a)
assert b == 3
c = a.sum()
assert c == 3

# Module-function form for a different constructor with a fill value.
d = np.full((2, 2), 3)
e = np.sum(d)
assert e == 12
