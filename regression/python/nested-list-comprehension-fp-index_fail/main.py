# Non-vacuity counterpart of nested-list-comprehension-fp-index: the inner float
# element of a 2-D comprehension is read at a constant outer index >= 1 and used
# in FP arithmetic.  The read returns the genuine value (0.0), so an assertion on
# the wrong value must FAIL with a counterexample (proving the read is real, not
# vacuously passing and not crashing the SMT backend, #5129).

C = [[0.0 for _ in range(2)] for _ in range(2)]
x = C[1][0]
assert x - 1.0 == 0.0
