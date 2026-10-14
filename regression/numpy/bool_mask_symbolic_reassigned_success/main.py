import numpy as np

# Unlike the row-select path (a[mask] on a 2-D array, which folds the mask
# statically and therefore rejects reassignment to stay sound), the 1-D
# boolean-mask path reads the mask through a runtime loop, so it always
# observes whichever value is in effect at this point in the program -
# reassigning the mask variable beforehand is safe here.
a = np.array([10, 20, 30])
n = nondet_bool()
mask = np.array([True, False, False])
mask = np.array([n, True, False])
b = a[mask]

assert b[len(b) - 1] == 20
