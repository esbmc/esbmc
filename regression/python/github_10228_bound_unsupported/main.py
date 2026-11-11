# Issue #10228: 2 ** 63 does not fit a 64-bit int parameter
def g(k: int) -> int:
    return k


n = 62
m = 0
while m < 2:
    g(2 ** n)
    n += 1
    m += 1
