import numpy as np

# Mixed slice/index tuple indexing on a 3-D array (exactly one full-slice
# axis, the rest literal ints) is now supported instead of being rejected -
# see build_mixed_slice_tuple_select.
a = np.array([[[1, 2, 3, 4]]])
b = a[:, 0, 0]

assert b[0] == 1
