import numpy as np

# np.equal(...) bound to a variable, then that variable used as
# np.logical_not()'s argument, must be evaluated the same way as the nested
# form.
a = [1, 2, 3]
b = [1, 5, 3]
x = np.equal(a, b)
r = np.logical_not(x)
assert r[0] == False
assert r[1] == True
assert r[2] == False
