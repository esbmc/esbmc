import numpy as np


# Same shape as array_return_param_shape_transpose_success (a second,
# function-local numpy alias forces the callee's own body to run instead of
# the caller inlining the whole call away), but with a wrong assertion to
# verify the test detects incorrect values.
def transposed(a):
    import numpy as localnp
    return localnp.transpose(a)


x = np.array([[1, 2], [3, 4]])
y = transposed(x)
assert y[0][1] == 2  # Should be 3
