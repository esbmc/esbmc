import numpy as np

# np.greater(...) bound to a variable, then that variable used as
# np.where()'s condition, must be evaluated the same way as the nested form.
a = [1, 2, 3]
b = [1, 5, 3]
cond = np.greater(b, a)
r = np.where(cond, [10, 20, 30], [1, 2, 3])
assert r[0] == 1
assert r[1] == 20
assert r[2] == 3
