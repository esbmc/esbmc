from decimal import Decimal

# 3.7 rounds to 4 (round half even)
d1: Decimal = Decimal("3.7")
r1: Decimal = d1.to_integral_value()
assert r1._int == 4
assert r1._exp == 0

# 3.5 rounds to 4 (odd quotient rounds up)
d2: Decimal = Decimal("3.5")
r2: Decimal = d2.to_integral_value()
assert r2._int == 4
assert r2._exp == 0

# 4.5 rounds to 4 (even quotient stays)
d3: Decimal = Decimal("4.5")
r3: Decimal = d3.to_integral_value()
assert r3._int == 4
assert r3._exp == 0

# 3.2 rounds to 3
d4: Decimal = Decimal("3.2")
r4: Decimal = d4.to_integral_value()
assert r4._int == 3
assert r4._exp == 0

# Negative: -2.7 rounds to -3
d5: Decimal = Decimal("-2.7")
r5: Decimal = d5.to_integral_value()
assert r5._sign == 1
assert r5._int == 3
assert r5._exp == 0

# Already integer: 42 stays 42
d6: Decimal = Decimal("42")
r6: Decimal = d6.to_integral_value()
assert r6._int == 42
assert r6._exp == 0

# Infinity passes through
d7: Decimal = Decimal("Infinity")
r7: Decimal = d7.to_integral_value()
assert r7._is_special == 1

# NaN passes through
d8: Decimal = Decimal("NaN")
r8: Decimal = d8.to_integral_value()
assert r8._is_special == 2
