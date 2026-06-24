from decimal import Decimal

# copy_abs
d: Decimal = Decimal("-3.14")
r: Decimal = d.copy_abs()
assert r._sign == 0
assert r._int == 314
assert r._exp == -2

# copy_negate
r2: Decimal = d.copy_negate()
assert r2._sign == 0
assert r2._int == 314

five: Decimal = Decimal("5")
r3: Decimal = five.copy_negate()
assert r3._sign == 1
assert r3._int == 5

# copy_sign
a: Decimal = Decimal("10")
b: Decimal = Decimal("-7")
r4: Decimal = a.copy_sign(b)
assert r4._sign == 1
assert r4._int == 10

# adjusted
d2: Decimal = Decimal("123")
assert d2.adjusted() == 2
d3: Decimal = Decimal("1.23")
assert d3.adjusted() == 0
d4: Decimal = Decimal("0.00123")
assert d4.adjusted() == -3

# is_normal / is_subnormal
d5: Decimal = Decimal("1.5")
assert d5.is_normal() == True
assert d5.is_subnormal() == False
d6: Decimal = Decimal("0")
assert d6.is_normal() == False
assert d6.is_subnormal() == False

# compare
one: Decimal = Decimal("1")
two: Decimal = Decimal("2")
c1: Decimal = one.compare(two)
assert c1._sign == 1
assert c1._int == 1
c2: Decimal = two.compare(one)
assert c2._sign == 0
assert c2._int == 1
twoB: Decimal = Decimal("2")
c3: Decimal = two.compare(twoB)
assert c3._int == 0

# max / min
three: Decimal = Decimal("3")
fivev: Decimal = Decimal("5")
m1: Decimal = three.max(fivev)
assert m1._int == 5
m2: Decimal = three.min(fivev)
assert m2._int == 3
