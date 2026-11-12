def f(x: int) -> int:
    if x > 0:
        assert x != 1
    return x


print(f(nondet_int()))
