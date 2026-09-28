import numpy as np

# a[i, :] (row-select-via-tuple, NOT the column-select shape a[:, j]) must
# keep working: is_column_select_slice_node in converter_stmt.cpp must not
# also match this reversed axis order, or the cached-RHS rebuild meant for
# column views forces current_lhs to be retyped mid-chain by
# try_build_row_pointer_view before the outer `[:]` is ever applied.
a = np.array([[1, 2], [3, 4], [5, 6]])
b = a[1, :]

assert b[0] == 3
assert b[1] == 4
