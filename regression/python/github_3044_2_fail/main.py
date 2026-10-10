def foo() -> tuple[int, int]:
    return (1, 2)

t: tuple[int, int] = foo()
(x, y) = t
if nondet_bool():
    assert x == 2
if nondet_bool():
    assert y == 1
