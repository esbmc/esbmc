def func() -> int:
    return 2


x: int = func()
y: float = 1 / x

assert y >= 0.5
