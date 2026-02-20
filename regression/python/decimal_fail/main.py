from decimal import Decimal

x: Decimal = Decimal("10.5")
assert x._sign == 1
