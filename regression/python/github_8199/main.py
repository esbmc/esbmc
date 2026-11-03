# Issue #8199: membership on a range with non-constant bounds.
def f(a: int, b: int) -> None:
    assert 5 in range(a, b)
    assert a in range(a, b)
    assert 0 not in range(a, b)
    assert b not in range(a, b)
    assert 7 in range(a, b, 3)
    assert 8 not in range(a, b, 3)
    assert 4 in range(b, a, -2)
    assert 9 in range(b)


f(1, 10)
