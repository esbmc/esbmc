import numpy as np

calls = 0


# Not a single return expression: the frontend treats the call as an escape.
def register(v):
    global calls
    calls = calls + 1


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
register(row)
assert row.shape == (3,)
assert row.ndim == 1
assert row.size == 3
assert len(row) == 3
assert calls == 1
