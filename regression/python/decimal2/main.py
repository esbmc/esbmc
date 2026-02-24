from decimal import Decimal

# Same value, different representation
a: Decimal = Decimal("1.0")
b: Decimal = Decimal("1.00")
assert a == b

# Simple equality
c: Decimal = Decimal("3.14")
d: Decimal = Decimal("3.14")
assert c == d

# Inequality
e: Decimal = Decimal("1.5")
f: Decimal = Decimal("2.5")
assert not (e == f)

# Zero with different signs
g: Decimal = Decimal("0")
h: Decimal = Decimal("-0")
assert g == h

# Integer construction equality
i: Decimal = Decimal(42)
j: Decimal = Decimal("42")
assert i == j
