# x ** 64 is 2**64 for x == 2, so the divisor is 0. In 64 bits it wraps to 0.
def p(x: int) -> int:
    return 100 // (x**64 // 2**62 - 4)


p(2)
