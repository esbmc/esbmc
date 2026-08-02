from decimal import Decimal

# int(10.5) truncates to 10, not 11
d: Decimal = Decimal("10.5")
assert int(d) == 11
