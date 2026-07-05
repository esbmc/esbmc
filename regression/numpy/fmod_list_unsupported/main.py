import numpy as np

# Dynamic Python lists passed to numpy.fmod (a broadcasting ufunc). ESBMC only
# models the scalar form, so container operands must be rejected cleanly rather
# than mis-folded to a scalar.
x = [5.5, 7.0]
y = [2.0, 3.0]
a = np.fmod(x, y)
assert a[0] == 1.5
