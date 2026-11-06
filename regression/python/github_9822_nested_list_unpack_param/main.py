def f(x):
    [a, b], c = x
    assert a < b
    return a + b + c


assert f(([1, 2], 3)) == 6
