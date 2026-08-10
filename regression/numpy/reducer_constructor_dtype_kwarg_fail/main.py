import numpy as np

# A dtype= keyword makes materialize_numpy_constructor_array() decline (it
# only handles bare constructor calls). That must not fall back to reading
# the shape/size argument as if it were the array's data -- eye(3)'s "3" is
# not the array, and its real mean (identity(3) flattened) is 1/3, not 3.
a = np.eye(3, dtype=int)
b = np.mean(a)

assert b == 3
