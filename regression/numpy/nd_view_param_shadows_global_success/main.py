import numpy as np


# The parameter shares the global's name: binding the global must not
# detach the function-local view of the parameter.
def first_col(a):
    v = a[:, 0]
    return v[1]


a = np.array([[1, 2], [3, 4]])
x = first_col(a)
assert x == 3
