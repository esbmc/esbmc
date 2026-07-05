# Issue #5115: symbolic numpy matmul must use floating-point arithmetic for
# float matrices. Before the fix the result element type was void (array-bounds
# violation) and float matmul dispatched to the integer dot(), reinterpreting
# the double bit pattern as int64 (unsound). It must now compute a * c.
import numpy as np

a = nondet_float()
c = nondet_float()
__ESBMC_assume(a >= 0.0 and a <= 5.0)
__ESBMC_assume(c >= 0.0 and c <= 5.0)

A = np.matmul([[a]], [[c]])
assert A[0][0] == a * c
