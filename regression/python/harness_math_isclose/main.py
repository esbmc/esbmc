# Verification harness for math.isclose (src/python-frontend/models/math.py).
#
# isclose(a, b, rel_tol, abs_tol) reports whether a and b are within the
# given tolerances: |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol).
#
# REQUIRES:
#   R1: x, y are finite nondet floats (bounded interval).
#
# ENSURES:
#   E1: isclose(x, x)                            [reflexive: every value is
#       close to itself]
#   E2: isclose(x, x, 0.0, 0.0)                  [reflexive even at zero
#       tolerance: |x - x| == 0 <= 0]
#   E3: not isclose(1.0, 2.0)                    [well-separated values at the
#       default tolerance are reported not close]
#
# A single nondet float drives E1/E2; E3 uses concrete anchors, avoiding the
# expensive two-symbolic-float tolerance product while still exercising the
# not-close branch.
import math

x: float = nondet_float()

__ESBMC_assume(x >= -1000.0)
__ESBMC_assume(x <= 1000.0)

assert math.isclose(x, x)  # E1
assert math.isclose(x, x, 0.0, 0.0)  # E2
assert not math.isclose(1.0, 2.0)  # E3
