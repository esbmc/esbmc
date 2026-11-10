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


assert floordiv(7, False) == -1
assert mod(7, False) == -2
assert quotient(7, False) == -3
assert float_mod(7.5, False) == -4.0
assert floordiv(7, True) == 7
assert mod(7, True) == 0
assert quotient(7, True) == 7
assert float_mod(7.5, True) == 0.5
