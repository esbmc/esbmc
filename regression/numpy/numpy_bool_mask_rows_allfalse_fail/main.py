import numpy as np

# ADR-NP-001: an all-false symbolic mask is still a symbolic (non-literal)
# mask, so it stays explicitly rejected the same as any other - there is no
# special-cased "zero selected" success path in the reverted model.
n = nondet_bool()
__ESBMC_assume(not n)

a = np.array([[1, 2], [3, 4]])
mask = np.array([n, n])
b = a[mask]
