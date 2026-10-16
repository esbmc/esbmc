def f(a: int) -> int:
    return a


x = f(1)
if x == 1:
    x = f()
