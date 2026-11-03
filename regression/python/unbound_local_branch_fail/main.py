def f(c: bool) -> int:
    if c:
        k = 1
    return k


f(nondet_bool())
