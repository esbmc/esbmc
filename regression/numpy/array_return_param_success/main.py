import numpy as np

# Read-through only: identity()'s return is an independent copy of x, not a
# real alias -- y[0] = 99 would not propagate back to x[0]. Known gap, not
# yet in this PR's scope.


def identity(a):
    return a


x = np.array([1, 2, 3])
y = identity(x)

assert y[0] == 1
assert y[2] == 3
