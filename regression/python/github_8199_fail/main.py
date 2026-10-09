# Issue #8199: the stop bound of a range is excluded.
def f(a: int, b: int) -> None:
    assert b in range(a, b)


f(1, 10)
