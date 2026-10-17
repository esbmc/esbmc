def f(n: int) -> int:
    x = n
    def bump() -> None:
        nonlocal x
        x += 10
    if n > 0:
        f(n - 1)
    bump()
    return x
assert f(2) == 12
