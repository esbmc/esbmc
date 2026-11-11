# Issue #10243: a False bool divisor raises a catchable ZeroDivisionError.


def floordiv(x: int, b: bool) -> int:
    try:
        return x // b
    except ZeroDivisionError:
        return -1


def mod(x: int, b: bool) -> int:
    try:
        return x % b
    except ZeroDivisionError:
        return -2


def quotient(x: int, b: bool) -> int:
    try:
        q, r = divmod(x, b)
        return q
    except ZeroDivisionError:
        return -3


def float_mod(y: float, b: bool) -> float:
    try:
        return y % b
    except ZeroDivisionError:
        return -4.0


def truediv(x: int, b: bool) -> float:
    try:
        return x / b
    except ZeroDivisionError:
        return -5.0


def float_floordiv(y: float, b: bool) -> float:
    try:
        return y // b
    except ZeroDivisionError:
        return -6.0


def float_quotient(y: float, b: bool) -> float:
    try:
        q, r = divmod(y, b)
        return q
    except ZeroDivisionError:
        return -7.0


def bool_floordiv(a: bool, b: bool) -> int:
    try:
        return a // b
    except ZeroDivisionError:
        return -8


def bool_quotient(a: bool, b: bool) -> int:
    try:
        q, r = divmod(a, b)
        return q
    except ZeroDivisionError:
        return -9


def aug_floordiv(x: int, b: bool) -> int:
    try:
        x //= b
        return x
    except ZeroDivisionError:
        return -10


def bool_float_quotient(a: bool, y: float) -> float:
    try:
        q, r = divmod(a, y)
        return q
    except ZeroDivisionError:
        return -11.0


calls = 0


def flag(b: bool) -> bool:
    global calls
    calls += 1
    return b


def call_quotient(x: int, b: bool) -> int:
    try:
        q, r = divmod(x, flag(b))
        return q
    except ZeroDivisionError:
        return -12


assert floordiv(7, False) == -1
assert mod(7, False) == -2
assert quotient(7, False) == -3
assert float_mod(7.5, False) == -4.0
assert truediv(7, False) == -5.0
assert float_floordiv(7.5, False) == -6.0
assert float_quotient(7.5, False) == -7.0
assert bool_floordiv(True, False) == -8
assert bool_quotient(True, False) == -9
assert aug_floordiv(7, False) == -10
assert floordiv(7, True) == 7
assert mod(7, True) == 0
assert quotient(7, True) == 7
assert float_mod(7.5, True) == 0.5
assert truediv(7, True) == 7.0
assert float_floordiv(7.5, True) == 7.0
assert float_quotient(7.5, True) == 7.0
assert bool_floordiv(True, True) == 1
assert bool_quotient(True, True) == 1
assert aug_floordiv(7, True) == 7
assert bool_float_quotient(True, 0.0) == -11.0
assert bool_float_quotient(True, 2.0) == 0.0
assert call_quotient(7, False) == -12
assert call_quotient(7, True) == 7
assert calls == 2
