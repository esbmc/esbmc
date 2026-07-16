import numpy as np

# Boolean-mask indexing through a parameter (a[mask]) is a separate,
# still-unsupported case - only direct/literal-index parameter usage is
# handled here, so `a` keeps its old Any/PyListObject* default and the call
# boundary rejects it explicitly instead of miscompiling.
def select(a, mask):
    return a[mask]

a = np.array([1, 2, 3])
mask = np.array([True, False, True])
b = select(a, mask)
