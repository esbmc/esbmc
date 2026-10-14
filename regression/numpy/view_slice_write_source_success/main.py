import numpy as np

# ADR-NP-003 etapa 2: a 1-D slice view now shares the base array's buffer,
# so writing through the base array is observed by a live view, instead of
# being rejected as an unsupported write to an array with a live view.
a = np.array([1, 2, 3, 4])
part = a[1:3]
a[1] = 10

assert part[0] == 10
