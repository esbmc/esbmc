from decimal import Decimal

# 10 - Decimal("3") = 7, not 3
r: Decimal = 10 - Decimal("3")
assert r._int == 3
