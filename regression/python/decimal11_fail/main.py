from decimal import Decimal

# "1.200" normalizes to (0, 12, -1, 0), NOT (0, 1200, -3, 0)
d: Decimal = Decimal("1.200")
n: Decimal = d.normalize()
assert n._int == 1200
