# Falsification harness for int.bit_count (src/python-frontend/models/int.py).
#
# Validates that a violated postcondition is detected: it conflates bit_count
# (population count) with bit_length (bit width), which are equal only when
# every bit up to the top is set (n == 2**k - 1).
#
# REQUIRES:
#   R1: n is a bounded non-negative integer.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: n.bit_count() == n.bit_length().  False e.g. for n == 2 (bit_count 1,
#       bit_length 2).
n: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 255)

assert n.bit_count() == n.bit_length()     # F1 — falsifiable
