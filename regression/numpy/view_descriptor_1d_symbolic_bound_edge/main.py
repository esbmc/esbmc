import numpy as np

# A slice with a symbolic (non-literal) bound cannot get a compile-time
# offset for the pointer-based descriptor this PR adds (ndarray_descriptor
# stores offset as a plain long long, not a symbolic expression), so it
# must keep working via the existing copy path -- unchanged behavior, no
# new diagnostic, and no upgrade to real aliasing for this case.
def f(n):
    a = np.array([1, 2, 3, 4])
    v = a[n : n + 2]
    return v[0]


r = f(1)
assert r == 2
