def f(n: int) -> float:
    a = n
    assert isinstance(a, int)
    b = a * 2
    a = a / 2
    assert isinstance(a, float)
    return a + b


assert f(4) == 10.0
