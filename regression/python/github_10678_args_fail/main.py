# t(2) calls 0 when flag is false, which raises TypeError.
def inc(x: int) -> int:
    return x + 1


def zero():
    return 0


def pick(flag):
    if flag:
        f = inc
    else:
        f = zero()
    return f


t = pick(nondet_bool())
assert t(2) == 3
