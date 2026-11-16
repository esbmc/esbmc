# x // -1 is 2**63 for x == -2**63, so the divisor is 0. In C++, -2**63 / -1
# overflows.
def q() -> int:
    x = -9223372036854775807 - 1
    return 100 // (x // -1 // 2**62 - 2)


q()
