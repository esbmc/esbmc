def check(v: int) -> int:
    if v > 0:
        return 1
    return 2


v: int = nondet_int()
check(v)
