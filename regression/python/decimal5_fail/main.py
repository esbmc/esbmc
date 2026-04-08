from decimal import Decimal

def check_is_nan(d: Decimal) -> bool:
    return d.is_nan()

# 3.14 is not NaN
a: Decimal = Decimal("3.14")
assert check_is_nan(a)
