import numpy as np

# An unsupported keyword (order=) makes materialize_numpy_constructor_array()
# decline for a dynamic-list-backed constructor (eye/identity/full/
# linspace). That is not a dimensionality problem, so the diagnostic must
# say so instead of the generic "up to 2D arrays" message used for an
# actual 3D+/non-rectangular shape. dtype= is a supported keyword now, so
# it no longer exercises this.
a = np.eye(3, order='C')
b = a.transpose()

assert b[0][0] == 1
