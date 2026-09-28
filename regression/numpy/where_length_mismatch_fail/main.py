import numpy as np

# np.where() with a condition list longer than one of its choice lists must
# raise an explicit diagnostic instead of an out-of-bounds JSON access when
# the shorter choice list is indexed past its end.
cond = [True, False, True]
x = [1, 2]
y = [10, 20, 30]
r = np.where(cond, x, y)
assert len(r) == 3
