import numpy as np

# numpy.fmod is a broadcasting ufunc, but ESBMC only models the scalar form.
# Array operands must be rejected with a clear diagnostic, not mis-folded to a
# scalar (which previously produced a wrong result / backend crash).
a = np.fmod(np.array([5.5, 7.0]), np.array([2.0, 3.0]))
assert a[0] == 1.5
