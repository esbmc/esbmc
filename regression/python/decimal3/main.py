from decimal import Decimal

# Basic ordering
a: Decimal = Decimal("1.5")
b: Decimal = Decimal("2.5")
assert a < b
assert b > a
assert a <= b
assert b >= a

# Equal values
c: Decimal = Decimal("3.0")
d: Decimal = Decimal("3.00")
assert c <= d
assert c >= d
assert not (c < d)
assert not (c > d)

# Negative numbers
e: Decimal = Decimal("-5")
f: Decimal = Decimal("-2")
assert e < f
assert f > e

# Negative vs positive
g: Decimal = Decimal("-1")
h: Decimal = Decimal("1")
assert g < h
assert h > g

# Zero comparisons
i: Decimal = Decimal("0")
j: Decimal = Decimal("-0")
assert not (i < j)
assert not (j < i)
assert i <= j
assert j >= i
