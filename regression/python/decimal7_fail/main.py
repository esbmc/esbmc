from decimal import Decimal

# zero should be falsy, so this branch should NOT be taken
d: Decimal = Decimal("0")
x: int = 0
if d:
    x = 1
assert x == 1
