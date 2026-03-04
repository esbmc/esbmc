from decimal import Decimal

def get_int(d: Decimal) -> int:
    return d._int

# 1.5 + 2.5 = 4.0, not 5.0
a: Decimal = Decimal("1.5")
b: Decimal = Decimal("2.5")
result: Decimal = a + b
assert get_int(result) == 50
