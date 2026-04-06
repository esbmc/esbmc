from decimal import Decimal

# Decimal.from_float(0.1) should have sign=0, is_special=0
d1: Decimal = Decimal.from_float(0.1)
assert d1._sign == 0
assert d1._is_special == 0

# Decimal.from_float(0.0) should be zero
d2: Decimal = Decimal.from_float(0.0)
assert d2._int == 0
assert d2._is_special == 0

# Decimal.from_float(1.0) = Decimal("1")
d3: Decimal = Decimal.from_float(1.0)
assert d3._sign == 0
assert d3._is_special == 0

# Decimal.from_float(-3.14) should be negative
d4: Decimal = Decimal.from_float(-3.14)
assert d4._sign == 1
assert d4._is_special == 0
