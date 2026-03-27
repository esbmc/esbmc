def foo() -> tuple[int, int]:
    return (1, 2)


t: tuple[int, int] = foo()
(x, y) = t
assert x == 2
assert y == 1
