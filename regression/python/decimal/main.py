from decimal import Decimal

x: Decimal = Decimal("10.5")
assert x._sign == 0
assert x._int == 105
assert x._exp == -1
assert x._is_special == 0

y: Decimal = Decimal("3")
assert y._sign == 0
assert y._int == 3
assert y._exp == 0

z: Decimal = Decimal(-42)
assert z._sign == 1
assert z._int == 42
assert z._exp == 0

w: Decimal = Decimal()
assert w._sign == 0
assert w._int == 0
assert w._exp == 0
assert w._is_special == 0
