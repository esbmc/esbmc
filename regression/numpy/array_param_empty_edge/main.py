import numpy as np

# Edge: an empty array passed to a function that never reads its parameter
# still gets a concrete (zero-length) array type at the boundary instead of
# tripping the lowering - the shape inference doesn't depend on whether the
# body actually uses the argument.
def unused(a) -> int:
    return 1

a = np.array([])
assert unused(a) == 1
