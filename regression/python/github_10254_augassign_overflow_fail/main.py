# d *= 2**62 makes d 2**64 for x == 4, so the divisor is 0. In 64 bits it wraps.
def g(x: int) -> int:
    d = x
    d *= 2**62
    return 100 // (d // 2**62 - x)


g(4)
