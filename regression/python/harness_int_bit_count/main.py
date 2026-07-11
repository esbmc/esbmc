# Verification harness for int.bit_count (src/python-frontend/models/int.py).
#
# n.bit_count() returns the number of one-bits in the binary representation
# of abs(n) (population count, Python 3.10+).
#
# REQUIRES:
#   R1: n is a bounded non-negative integer, so the shift loop terminates
#       well before the model's 512-bit literal cap.
#
# ENSURES (for c = n.bit_count()):
#   E1: c >= 0                                    [a count is non-negative]
#   E2: c <= n.bit_length()                       [at most one 1 per bit]
#   E3: (c == 0) == (n == 0)                       [only 0 has no set bits]
n: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 255)

c: int = n.bit_count()

assert c >= 0  # E1
assert c <= n.bit_length()  # E2
assert (c == 0) == (n == 0)  # E3

# Concrete anchors: powers of two have one bit, all-ones have full count.
assert (0).bit_count() == 0
assert (1).bit_count() == 1
assert (5).bit_count() == 2
assert (255).bit_count() == 8
