def f(n: int) -> None:
    x = n * 0.5
    assert x == 1.5


def g(n: int):
    return n * 0.5


f(3)
assert g(3) == 1.5
