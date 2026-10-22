import numpy as np

def apply_fn(row, fn):
    return fn(row)

def run():
    a = np.array([2, 4, 6])
    v = a[0:]
    return apply_fn(v, lambda x: x[0])

result = run()
assert result == 2
