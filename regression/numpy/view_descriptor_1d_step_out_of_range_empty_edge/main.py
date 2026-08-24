import numpy as np

# a[-10:1:-1] resolves to zero elements: the negative-step start clamps to
# -1 (literal_slice_length's "no elements" sentinel), which must not be
# used to form a pointer one element before the source's buffer (only a
# one-past-the-end pointer is legal to form without dereferencing it).
# Guards the code-review-found UB in try_build_1d_pointer_view.
a = np.array([1, 2, 3])
v = a[-10:1:-1]

assert len(v) == 0
