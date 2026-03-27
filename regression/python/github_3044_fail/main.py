def foo() -> tuple[int, int]:
    return (1, 2)


x: int
y: int
x, y = foo()

assert x == 2
assert y == 1
