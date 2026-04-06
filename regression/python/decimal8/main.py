from decimal import Decimal

# int() on whole number
d1: Decimal = Decimal("42")
assert int(d1) == 42

# int() on negative
d2: Decimal = Decimal("-7")
assert int(d2) == -7

# int() with positive exponent
d3: Decimal = Decimal(0, 5, 2, 0)
assert int(d3) == 500

# int() truncates fractional part
d4: Decimal = Decimal("10.5")
assert int(d4) == 10

# int() on zero
d5: Decimal = Decimal("0")
assert int(d5) == 0
