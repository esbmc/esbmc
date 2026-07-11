# Verification harness for decimal.Decimal arithmetic
# (src/python-frontend/models/decimal.py).
#
# The Decimal model implements fixed-point arithmetic (sign / coefficient /
# exponent). Decimal() only accepts constant arguments, so the harness uses
# concrete literals and checks algebraic identities that must hold for any
# faithful arithmetic.
#
# ENSURES (a = 1.5, b = 2.5):
#   E1: addition and multiplication commute
#   E2: 0 is the additive identity and a - a == 0
#   E3: 1 is the multiplicative identity
#   E4: negation is an involution and abs is sign-independent
#   E5: a concrete sum has the expected value
from decimal import Decimal

a: Decimal = Decimal("1.5")
b: Decimal = Decimal("2.5")
zero: Decimal = Decimal("0")
one: Decimal = Decimal("1")

assert a + b == b + a           # E1
assert a * b == b * a           # E1
assert a + zero == a            # E2
assert a - a == zero            # E2
assert a * one == a            # E3
assert -(-a) == a               # E4
assert abs(-a) == abs(a)        # E4
assert a + b == Decimal("4.0")  # E5
