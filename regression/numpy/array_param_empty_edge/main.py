import numpy as np

# Edge: the rejection is purely type-based at the call boundary, so even an
# empty array passed to a function that never reads its parameter is
# rejected the same way - the boundary stays predictable regardless of
# array contents or whether the body actually uses the argument.
def unused(a) -> int:
    return 1

a = np.array([])
assert unused(a) == 1
