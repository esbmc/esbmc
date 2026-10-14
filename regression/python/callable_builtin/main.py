def f() -> int:
    return 1
x: int = 5
assert callable(f)
assert callable(abs)
assert not callable(x)
