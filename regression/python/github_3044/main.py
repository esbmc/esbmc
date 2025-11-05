def foo() -> tuple[int, int]:
    return (1, 2)

x: int
y: int
x, y = foo()

assert x == 1
assert y == 2
