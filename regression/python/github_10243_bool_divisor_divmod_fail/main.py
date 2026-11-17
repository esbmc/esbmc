# Issue #10243: divmod(x, False) raises ZeroDivisionError.


def f(x: int, b: bool) -> int:
    q, r = divmod(x, b)
    return q


f(7, False)
