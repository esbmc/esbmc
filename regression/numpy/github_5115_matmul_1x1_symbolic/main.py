"""
Issue #5115: numpy.matmul/dot on symbolic inputs
Minimal reproducer: 1x1 matrix multiplication with symbolic elements
Should produce correct result without array bounds violation
"""
import numpy as np

a = nondet_int()
c = nondet_int()
__ESBMC_assume(a >= -5 and a <= 5)
__ESBMC_assume(c >= -5 and c <= 5)

# 1x1 @ 1x1 matrix multiplication
A = np.matmul([[a]], [[c]])

# Verify the result
assert A[0][0] == a * c
