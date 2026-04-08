from decimal import Decimal

# float(Decimal("42")) is 42.0, NOT 0.0
d: Decimal = Decimal("42")
f: float = float(d)
assert f == 0.0
