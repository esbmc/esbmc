def f(a: int) -> int:
    return a


if nondet_bool():
    x = f(a=1)
else:
    x = f(b=2)
