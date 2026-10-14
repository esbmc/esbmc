# bool() of an int expression or a call converts its value (x != 0).
def g(x: float) -> float:
    return x * 0.5


def h(a: int) -> int:
    return a


k: int = -1
b = bool(~k)
assert not b
assert bool(~k) == False
x: int = 7
assert bool(x & 8) == False
assert bool(g(1.0)) == True
assert bool(h(5)) + 1 == 2
