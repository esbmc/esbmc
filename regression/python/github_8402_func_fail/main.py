def f(n: int) -> float:
    a = n
    b = a * 2
    assert isinstance(a, float)
    a = a / 2
    return a + b


f(4)
