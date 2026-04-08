from decimal import Decimal

# Reverse add: 3 + Decimal("1.5") = 4.5
d: Decimal = Decimal("1.5")
r1: Decimal = 3 + d
assert r1._int == 45
assert r1._exp == -1

# Reverse sub: 10 - Decimal("3") = 7
r2: Decimal = 10 - Decimal("3")
assert r2._int == 7
assert r2._sign == 0

# Reverse mul: 4 * Decimal("2.5") = 10.0
r3: Decimal = 4 * Decimal("2.5")
assert r3._int == 100
assert r3._exp == -1
