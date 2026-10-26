import numpy as np

a = np.array([1, 2, 3])
b = np.zeros(3, dtype=int)
try:
    for x, y in np.nditer([a, b], op_flags=[["readonly"], ["readwrite"]]):
        y[...] = x
        if x == 2:
            raise ValueError("stop")
except ValueError:
    pass
assert b[0] == 1
assert b[1] == 2
assert b[2] == 0

flags_error = False
try:
    np.nditer(a, flags=["c_index", "f_index"])
except ValueError:
    flags_error = True
assert flags_error
