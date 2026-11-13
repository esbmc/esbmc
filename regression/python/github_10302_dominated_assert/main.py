def f(x: int) -> int:
    if x > 0:
        assert x != 0
    return x


print(f(nondet_int()))
