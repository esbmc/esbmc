def f(k: int) -> int:
    g = lambda x: x + k
    return g(2)


assert f(1) == 3
assert f(5) == 7
