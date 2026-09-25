import numpy as np

# mask passes through one intermediate function
# (forwarded through another function's own array/mask parameters).
def select_first_row(a, mask):
    b = a[mask]
    return b[0][0] * 10 + b[0][1]

def wrapper(a, mask):
    return select_first_row(a, mask)

a = np.array([[1, 2], [3, 4], [5, 6]])
m0 = nondet_bool()
m1 = nondet_bool()
m2 = nondet_bool()
__ESBMC_assume(not m0)
__ESBMC_assume(m1)
__ESBMC_assume(not m2)
mask = np.array([m0, m1, m2])
result = wrapper(a, mask)

assert result == 34
