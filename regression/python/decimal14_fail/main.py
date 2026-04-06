from decimal import Decimal

# "2.5" quantized to "1" rounds to 2 (half-even), NOT 3
d: Decimal = Decimal("2.5")
q: Decimal = Decimal("1")
r: Decimal = d.quantize(q)
assert r._int == 3
