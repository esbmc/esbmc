def f(x: int) -> None:
    if x > 0:
        return
    return

r = f(5)
assert (r is None)
