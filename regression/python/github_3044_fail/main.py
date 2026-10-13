def foo() -> tuple[int, int]:
    return (1, 2)

x: int
y: int
x, y = foo()

if nondet_bool():
    assert x == 2
if nondet_bool():
    assert y == 1
