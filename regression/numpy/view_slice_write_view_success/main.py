import numpy as np

# ADR-NP-003 etapa 2: a 1-D slice view now shares the base array's buffer,
# so writing through the view mutates the base array too, instead of being
# rejected as an unsupported write-through-a-copy.
a = np.array([1, 2, 3, 4])
part = a[1:3]
part[0] = 10

assert a[1] == 10
assert part[0] == 10
