from decimal import Decimal

# "3.14159" quantized to "0.01" -> (0, 314, -2, 0)
d1: Decimal = Decimal("3.14159")
q1: Decimal = Decimal("0.01")
r1: Decimal = d1.quantize(q1)
assert r1._int == 314
assert r1._exp == -2

# "5" quantized to "0.01" -> (0, 500, -2, 0)
d2: Decimal = Decimal("5")
r2: Decimal = d2.quantize(q1)
assert r2._int == 500
assert r2._exp == -2

# "2.5" quantized to "1" -> (0, 2, 0, 0) (half-even: even stays)
d3: Decimal = Decimal("2.5")
q2: Decimal = Decimal("1")
r3: Decimal = d3.quantize(q2)
assert r3._int == 2
assert r3._exp == 0

# "3.5" quantized to "1" -> (0, 4, 0, 0) (half-even: odd rounds up)
d4: Decimal = Decimal("3.5")
r4: Decimal = d4.quantize(q2)
assert r4._int == 4
assert r4._exp == 0

# Negative preserves sign
d5: Decimal = Decimal("-1.75")
q3: Decimal = Decimal("0.1")
r5: Decimal = d5.quantize(q3)
assert r5._sign == 1
assert r5._int == 18
assert r5._exp == -1

# Inf + Inf -> Inf
d6: Decimal = Decimal("Infinity")
q_inf: Decimal = Decimal("Infinity")
r6: Decimal = d6.quantize(q_inf)
assert r6._is_special == 1

# NaN propagation
d7: Decimal = Decimal("NaN")
q_one: Decimal = Decimal("1")
r7: Decimal = d7.quantize(q_one)
assert r7._is_special == 2
