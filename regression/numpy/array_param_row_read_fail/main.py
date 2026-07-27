import numpy as np

# The array parameter itself stays usable inside the callee (see
# numpy_param_array_success), but returning a whole row *out* of a function
# is a separate, still-unsupported case (arrays aren't valid by-value return
# types in the current model) - so `row` here decays to a scalar default.
def get_row(a):
    row = a[0]
    return row

a = np.array([[1, 2], [3, 4]])
row = get_row(a)
assert row[0] == 1
assert row[1] == 2
