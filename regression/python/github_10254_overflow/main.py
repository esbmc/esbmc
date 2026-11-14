# Issue #10254: a call whose 64-bit arithmetic does not wrap is still folded.
def w(x: int) -> int:
    d = (x * 2**61) // 2**61 - x + 1
    return 100 // d


def g(a: int) -> int:
    a //= 2
    a *= 3
    return a


assert w(3) == 100
assert g(-7) == -12
