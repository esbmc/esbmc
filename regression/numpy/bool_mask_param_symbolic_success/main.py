import numpy as np

# a[mask] with both a and mask received as parameters,
# mask symbolic (nondet-derived). Returning the row-selection result itself
# is a separate, still-unsupported case (arrays aren't valid by-value
# return types yet), so the callee reads it directly instead.
def select_first_row(a, mask):
    b = a[mask]
    assert b.shape[0] == 1
    return b[0][0] * 10 + b[0][1]

a = np.array([[1, 2], [3, 4], [5, 6]])
m0 = nondet_bool()
m1 = nondet_bool()
m2 = nondet_bool()
__ESBMC_assume(not m0)
__ESBMC_assume(m1)
__ESBMC_assume(not m2)
mask = np.array([m0, m1, m2])
result = select_first_row(a, mask)

assert result == 34
