import numpy as np

# numpy.reshape(a, newshape, order='C') has no split-dimension form: a third
# positional argument is `order`, not another dimension. Only the method
# form a.reshape(d1, d2, ...) accepts dimensions split across arguments.
a = np.array([1, 2, 3, 4, 5, 6])
b = np.reshape(a, 2, 3)
assert b[0][0] == 1
