from decimal import Decimal


def get_sign(d: Decimal) -> int:
    return d._sign


def get_int(d: Decimal) -> int:
    return d._int


def get_exp(d: Decimal) -> int:
    return d._exp


def get_is_special(d: Decimal) -> int:
    return d._is_special


x: Decimal = Decimal("10.5")
assert get_sign(x) == 0
assert get_int(x) == 105
assert get_exp(x) == -1
assert get_is_special(x) == 0

y: Decimal = Decimal("3")
assert get_sign(y) == 0
assert get_int(y) == 3
assert get_exp(y) == 0

z: Decimal = Decimal(-42)
assert get_sign(z) == 1
assert get_int(z) == 42
assert get_exp(z) == 0

w: Decimal = Decimal()
assert get_sign(w) == 0
assert get_int(w) == 0
assert get_exp(w) == 0
assert get_is_special(w) == 0
