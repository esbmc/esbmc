# Verification harness for int.bit_length (src/python-frontend/models/int.py).
#
# n.bit_length() returns the number of bits needed to represent abs(n) in
# binary, excluding sign and leading zeros (0 has bit_length 0).
#
# REQUIRES:
#   R1: n is a bounded non-negative integer, so the shift loop terminates
#       well before the model's 512-bit literal cap.
#
# ENSURES (for r = n.bit_length()):
#   E1: r >= 0                                    [bit count is non-negative]
#   E2: n < (1 << r)                              [r bits suffice to hold n]
#   E3: n == 0 or (1 << (r - 1)) <= n             [r is minimal: the top bit
#       is set, so r-1 bits would not suffice]
#
# E2 and E3 together pin r to the exact value and rule out off-by-one.
n: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 255)

r: int = n.bit_length()

assert r >= 0  # E1
assert n < (1 << r)  # E2
assert n == 0 or (1 << (r - 1)) <= n  # E3

# Concrete anchors for the boundary values.
assert (0).bit_length() == 0
assert (1).bit_length() == 1
assert (255).bit_length() == 8
