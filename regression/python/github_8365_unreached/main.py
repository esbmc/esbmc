def f(a: int) -> int:
    return a


x = f(1)
if x == 2:
    x = f()
assert x == 1
