from decimal import Decimal

# sqrt(-4) is NaN, asserting is_finite() should fail
d: Decimal = Decimal("-4")
s: Decimal = d.sqrt()
assert s.is_finite()
