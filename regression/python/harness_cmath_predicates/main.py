# Verification harness for cmath.isfinite / isnan / isinf
# (src/python-frontend/models/cmath.py).
#
# A complex value built from bounded (hence finite) components is finite:
# isfinite is True, isnan and isinf are False.
#
# REQUIRES:
#   R1: nondet real and imaginary parts, bounded to a finite interval.
#
# ENSURES (z = complex(a, b) with finite a, b):
#   E1: cmath.isfinite(z) is True
#   E2: cmath.isnan(z) is False
#   E3: cmath.isinf(z) is False
import cmath

a: float = nondet_float()
b: float = nondet_float()
__ESBMC_assume(-1000.0 <= a <= 1000.0)
__ESBMC_assume(-1000.0 <= b <= 1000.0)

z: complex = complex(a, b)

assert cmath.isfinite(z)        # E1
assert not cmath.isnan(z)       # E2
assert not cmath.isinf(z)       # E3
