# A call through a parameter takes its type from the functions every call
# site passes (#8351).
def apply(f, a: int) -> int:
    r = f(a)
    return r


def inc(a: int) -> int:
    return a + 1


def dbl(a: int) -> int:
    return a * 2


assert apply(inc, 1) == 2
assert apply(dbl, 3) == 7
