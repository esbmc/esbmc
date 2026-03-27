def f() -> float:
    op = lambda a, b: a + b
    return op(3.0, 5.0)

assert f() == 8.0
