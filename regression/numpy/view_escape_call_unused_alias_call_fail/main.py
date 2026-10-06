import numpy as np


def boom():
    raise ValueError("boom")


# The call bound to the unused name still runs, so the function cannot be
# reduced to its return expression.
def keep(v):
    t = boom()
    return v[1]


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
assert keep(row) == 2
