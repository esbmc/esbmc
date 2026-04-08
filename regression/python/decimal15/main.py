from decimal import Decimal

# sqrt(4) == 2 (use int() to check)
d1: Decimal = Decimal("4")
s1: Decimal = d1.sqrt()
assert int(s1) == 2

# sqrt(0) == 0
d2: Decimal = Decimal("0")
s2: Decimal = d2.sqrt()
assert s2._int == 0

# sqrt(-4) -> NaN
d3: Decimal = Decimal("-4")
s3: Decimal = d3.sqrt()
assert s3.is_nan()

# sqrt(Infinity) -> Infinity
d4: Decimal = Decimal("Infinity")
s4: Decimal = d4.sqrt()
assert s4.is_infinite()
assert s4._sign == 0

# sqrt(NaN) -> NaN
d5: Decimal = Decimal("NaN")
s5: Decimal = d5.sqrt()
assert s5.is_nan()

# sqrt(-Infinity) -> NaN
d6: Decimal = Decimal("-Infinity")
s6: Decimal = d6.sqrt()
assert s6.is_nan()

# sqrt(-0) preserves sign (returns -0)
d7: Decimal = Decimal("-0")
s7: Decimal = d7.sqrt()
assert s7._sign == 1
assert s7._int == 0
