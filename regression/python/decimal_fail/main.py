from decimal import Decimal


def get_sign(d: Decimal) -> int:
    return d._sign


x: Decimal = Decimal("10.5")
assert get_sign(x) == 1
