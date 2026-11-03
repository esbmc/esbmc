def g(xs: list[int]):
    yield 0
    yield from xs
    yield from range(3)


assert list(g([5, 6])) == [0, 5, 6, 0, 1]
