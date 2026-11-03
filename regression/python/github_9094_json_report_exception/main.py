def f(n: int) -> int:
    return 10 // n


x: int = nondet_int()
f(x)
