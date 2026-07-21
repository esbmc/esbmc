import numpy as np

# ADR-NP-001: a comparison-derived 2-D row mask is now supported (see
# numpy-architecture-decisions.md).
x = nondet_int()
y = nondet_int()
n = nondet_bool()
__ESBMC_assume(n == (x > y))
__ESBMC_assume(x > y)

a = np.array([[5, 6], [7, 8]])
mask = np.array([n, not n])
b = a[mask]
