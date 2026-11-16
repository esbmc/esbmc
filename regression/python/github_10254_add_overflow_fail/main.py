# x + 2**62 + 2**62 + 2**62 + 2**62 is 2**64 for x == 0, so d is 0. In 64 bits
# the sum wraps to 0 and d would be -4.
def a(x: int) -> int:
    d = (x + 2**62 + 2**62 + 2**62 + 2**62) // 2**62 - 4
    return 100 // d


a(0)
