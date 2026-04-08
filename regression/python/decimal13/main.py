from decimal import Decimal

# float(Decimal("42")) == 42.0
d1: Decimal = Decimal("42")
f1: float = float(d1)
assert f1 == 42.0

# float(Decimal("0.5")) == 0.5
d2: Decimal = Decimal("0.5")
f2: float = float(d2)
assert f2 == 0.5

# Negative: float(Decimal("-3.14")) < 0.0
d3: Decimal = Decimal("-3.14")
f3: float = float(d3)
assert f3 < 0.0

# Zero: float(Decimal("0")) == 0.0
d4: Decimal = Decimal("0")
f4: float = float(d4)
assert f4 == 0.0
