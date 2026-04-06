from decimal import Decimal

# "1.200" = (0, 1200, -3, 0) -> normalize -> (0, 12, -1, 0)
d1: Decimal = Decimal("1.200")
n1: Decimal = d1.normalize()
assert n1._int == 12
assert n1._exp == -1

# "100" = (0, 100, 0, 0) -> normalize -> (0, 1, 2, 0)
d2: Decimal = Decimal("100")
n2: Decimal = d2.normalize()
assert n2._int == 1
assert n2._exp == 2

# "0.00" = (0, 0, -2, 0) -> normalize -> (0, 0, 0, 0)
d3: Decimal = Decimal("0.00")
n3: Decimal = d3.normalize()
assert n3._int == 0
assert n3._exp == 0

# Negative: "-1.200" preserves sign
d4: Decimal = Decimal("-1.200")
n4: Decimal = d4.normalize()
assert n4._sign == 1
assert n4._int == 12
assert n4._exp == -1

# Already normalized: "3.14" = (0, 314, -2, 0) stays the same
d5: Decimal = Decimal("3.14")
n5: Decimal = d5.normalize()
assert n5._int == 314
assert n5._exp == -2

# Infinity passes through
d6: Decimal = Decimal("Infinity")
n6: Decimal = d6.normalize()
assert n6._is_special == 1

# NaN passes through (as qNaN)
d7: Decimal = Decimal("NaN")
n7: Decimal = d7.normalize()
assert n7._is_special == 2
