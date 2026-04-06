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

r3: Decimal = Decimal("5").copy_negate()
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
c1: Decimal = Decimal("1").compare(Decimal("2"))
assert c1._sign == 1
assert c1._int == 1
c2: Decimal = Decimal("2").compare(Decimal("1"))
assert c2._sign == 0
assert c2._int == 1
c3: Decimal = Decimal("1").compare(Decimal("1"))
assert c3._int == 0

# max / min
m1: Decimal = Decimal("3").max(Decimal("5"))
assert m1._int == 5
m2: Decimal = Decimal("3").min(Decimal("5"))
assert m2._int == 3

# fma: 2 * 3 + 4 = 10
f: Decimal = Decimal("2").fma(Decimal("3"), Decimal("4"))
assert f._int == 10
assert f._exp == 0
