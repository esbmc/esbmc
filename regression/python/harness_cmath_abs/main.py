# Verification harness for complex magnitude (abs) via the cmath model
# (src/python-frontend/models/cmath.py).
#
# abs(z) is the modulus sqrt(re^2 + im^2). In IEEE-754 the squared magnitude
# is NOT exactly re^2 + im^2 (sqrt rounds, and re^2 rounds first), so the only
# exact facts are non-negativity and the perfect-square anchors below —
# deliberately chosen because 25 and 169 have exact float square roots.
#
# REQUIRES:
#   R1: nondet real and imaginary parts, bounded to a finite interval.
#
# ENSURES (z = complex(a, b), r = abs(z)):
#   E1: the modulus is non-negative
#   E2: concrete Pythagorean triples give the exact hypotenuse
import cmath

a: float = nondet_float()
b: float = nondet_float()
__ESBMC_assume(-100.0 <= a <= 100.0)
__ESBMC_assume(-100.0 <= b <= 100.0)

z: complex = complex(a, b)
r: float = abs(z)

assert r >= 0.0                             # E1

assert abs(complex(0.0, 0.0)) == 0.0        # E2
assert abs(complex(3.0, 4.0)) == 5.0        # E2
assert abs(complex(5.0, 12.0)) == 13.0      # E2
