# Verification harness for int.conjugate (src/python-frontend/models/int.py).
#
# n.conjugate() returns n unchanged: the complex conjugate of a real integer
# is itself (int participates in the numeric-tower API).
#
# REQUIRES:
#   R1: n is a bounded integer (positive, negative and zero all covered).
#
# ENSURES:
#   E1: n.conjugate() == n                        [identity on reals]
#   E2: applying conjugate twice is still n       [involution]
n: int = nondet_int()

__ESBMC_assume(-1000 <= n <= 1000)

m: int = n.conjugate()
assert m == n                                  # E1
assert m.conjugate() == n                      # E2
