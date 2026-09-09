import numpy as np


def symbolic_shape(a):
    # This test is incomplete - just ensure it rejects
    return a.shape


# Trying to pass a symbolic array without concrete shape
a = np.array([1, 2, 3])
a_nondet = a  # nondeterministic shape
shape_result = symbolic_shape(a_nondet)

assert shape_result[0] == 3
