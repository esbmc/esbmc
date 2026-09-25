# bool() of a call inside a larger expression kept the call's int value.
def g(x: int) -> int:
    return x * 5


k: int = 1
if k > 0:
    k = k + 0
assert bool(g(k)) + 1 == 6
