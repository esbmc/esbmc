from decimal import Decimal

# __bool__ via if/else
d1: Decimal = Decimal("3.14")
x: int = 0
if d1:
    x = 1
else:
    x = 2
assert x == 1

# zero is falsy
d2: Decimal = Decimal("0")
y: int = 0
if d2:
    y = 1
else:
    y = 2
assert y == 2

# Infinity is truthy
d3: Decimal = Decimal("Infinity")
z: int = 0
if d3:
    z = 1
else:
    z = 2
assert z == 1

# NaN is truthy
d4: Decimal = Decimal("NaN")
w: int = 0
if d4:
    w = 1
else:
    w = 2
assert w == 1
