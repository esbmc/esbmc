import numpy as np

# A constructor call carrying an unsupported keyword argument (order=) is
# outside the conservative materialization this block adds; must keep the
# existing explicit diagnostic instead of silently misreading the shape as
# data. dtype= is a supported keyword now, so it no longer exercises this.
a = np.zeros((2, 2), order='C')
b = a.flatten()

assert b[0] == 0
