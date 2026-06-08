# Issue #5115 (negative): a wrong assertion on symbolic float matmul must yield
# a real VERIFICATION FAILED, proving the result is genuinely computed and the
# verdict is not vacuous.
import numpy as np

a = nondet_float()
c = nondet_float()
__ESBMC_assume(a >= 0.0 and a <= 5.0)
__ESBMC_assume(c >= 0.0 and c <= 5.0)

A = np.matmul([[a]], [[c]])
assert A[0][0] == a * c + 1.0
