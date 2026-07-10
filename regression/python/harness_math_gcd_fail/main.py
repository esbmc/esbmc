# Falsification harness for math.gcd (src/python-frontend/models/math.py).
#
# This negative harness validates that the verification machinery detects a
# violated postcondition: it asserts a plausible-but-WRONG strengthening of
# the gcd contract, so ESBMC must report VERIFICATION FAILED.
#
# REQUIRES:
#   R1: a, b bounded integers (as in harness_math_gcd).
#   R2: a and b not both zero.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: gcd(a, b) >= 2.  False whenever a and b are coprime, e.g. a == b == 1
#       gives gcd == 1.  A correct harness must catch this.
import math

a: int = nondet_int()
b: int = nondet_int()

__ESBMC_assume(a >= 0)
__ESBMC_assume(a <= 20)
__ESBMC_assume(b >= 0)
__ESBMC_assume(b <= 20)
__ESBMC_assume(a + b > 0)

g: int = math.gcd(a, b)

assert g >= 2            # F1 — falsifiable (coprime inputs give gcd == 1)
