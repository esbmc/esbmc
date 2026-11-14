# Issue #10243: x // False raises ZeroDivisionError, not a C division by zero.


def f(x: int, b: bool) -> int:
    return x // b


f(7, False)
