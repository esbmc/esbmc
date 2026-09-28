import numpy as np

# `a[1:3]` here is not itself the assignment's RHS -- it is the base of the
# outer `[0]` index. current_lhs must not leak into converting that base, or
# x (which should end up a plain scalar) gets wrongly retyped to a pointer
# for the intermediate slice value.
a = np.array([1, 2, 3, 4])
x = a[1:3][0]

assert x == 2
