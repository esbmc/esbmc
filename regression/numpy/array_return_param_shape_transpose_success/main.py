import numpy as np


# A second, function-local alias for numpy disables the caller-side
# return-value inlining that normally masks this gap (see test.desc):
# it forces `transposed`'s own body to actually run, instead of the
# caller substituting and folding the whole call away.
def transposed(a):
    import numpy as localnp
    return localnp.transpose(a)


x = np.array([[1, 2], [3, 4]])
y = transposed(x)
assert y[0][1] == 3
