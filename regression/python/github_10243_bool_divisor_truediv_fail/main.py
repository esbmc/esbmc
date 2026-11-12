# Issue #10243: x / False raises ZeroDivisionError.


def f(x: int, b: bool) -> float:
    return x / b


f(7, False)
