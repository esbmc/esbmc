import numpy as np

z = np.zeros((0,), dtype=np.int64)
with np.nditer(z, flags=["buffered", "zerosize_ok"], op_flags=[["readwrite"]],
               op_dtypes=["float64"], casting="unsafe") as it:
    for x in it:
        x[...] = 1.0
    assert it.finished
assert z.size == 0
