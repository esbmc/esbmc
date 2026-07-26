import numpy as np

# 3 indices on a genuinely 2-D array is an arity mismatch, not the general
# "multi-dimensional indexing is not supported" gap - now caught precisely
# by build_mixed_slice_tuple_select's own dimension walk.
a = np.array([[1, 2], [3, 4]])
x = a[:, 0, 1]
