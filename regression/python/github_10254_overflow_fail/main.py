# Issue #10254: folding w(4) in 64 bits wrapped and dropped this
# ZeroDivisionError.
def w(x: int) -> int:
    d = (x * 2**62) // 2**62 - x
    return 100 // d


w(4)
