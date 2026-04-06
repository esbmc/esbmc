from decimal import Decimal

# 3.5 rounds to 4 (half-even: odd quotient rounds up), NOT 3
d: Decimal = Decimal("3.5")
r: Decimal = d.to_integral_value()
assert r._int == 3
