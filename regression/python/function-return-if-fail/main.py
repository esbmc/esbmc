def func() -> int:
    return 1


x: int = func()
if (x == 0):
    y: float = 1 / x
else:
    assert (x == 0)
