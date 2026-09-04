counter: int = 1


def bump() -> None:
    global counter
    counter += 1


def f(n: int) -> None:
    c = [(1, 2)]
    k = lambda t: t[0]
    assert k(c[0]) == 2


f(3)
