from decimal import Decimal


def get_sign(d: Decimal) -> int:
    return d._sign


def get_int(d: Decimal) -> int:
    return d._int


def get_exp(d: Decimal) -> int:
    return d._exp


def get_is_special(d: Decimal) -> int:
    return d._is_special


# Negation
a: Decimal = Decimal("3.0")
neg_a: Decimal = -a
assert get_sign(neg_a) == 1
assert get_int(neg_a) == 30
assert get_exp(neg_a) == -1

# Addition: 1.5 + 2.5 = 4.0
b: Decimal = Decimal("1.5")
c: Decimal = Decimal("2.5")
result_add: Decimal = b + c
assert get_sign(result_add) == 0
assert get_int(result_add) == 40
assert get_exp(result_add) == -1

# Subtraction: 5.0 - 3.0 = 2.0
d: Decimal = Decimal("5.0")
e: Decimal = Decimal("3.0")
result_sub: Decimal = d - e
assert get_sign(result_sub) == 0
assert get_int(result_sub) == 20
assert get_exp(result_sub) == -1

# Multiplication: 2 * 3 = 6
f: Decimal = Decimal("2")
g: Decimal = Decimal("3")
result_mul: Decimal = f * g
assert get_sign(result_mul) == 0
assert get_int(result_mul) == 6
assert get_exp(result_mul) == 0

# Floor division: 7 // 2 = 3
h: Decimal = Decimal("7")
i: Decimal = Decimal("2")
result_floordiv: Decimal = h // i
assert get_sign(result_floordiv) == 0
assert get_int(result_floordiv) == 3
assert get_exp(result_floordiv) == 0

# Modulo: 7 % 2 = 1
result_mod: Decimal = h % i
assert get_sign(result_mod) == 0
assert get_int(result_mod) == 1
assert get_exp(result_mod) == 0

# Subtraction resulting in negative: 2 - 5 = -3
result_neg_sub: Decimal = f - d
assert get_sign(result_neg_sub) == 1
assert get_int(result_neg_sub) == 30
assert get_exp(result_neg_sub) == -1
