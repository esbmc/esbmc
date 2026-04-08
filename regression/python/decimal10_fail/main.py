from decimal import Decimal

# Decimal.from_float(-3.14) is negative, asserting sign==0 should fail
d: Decimal = Decimal.from_float(-3.14)
assert d._sign == 0
