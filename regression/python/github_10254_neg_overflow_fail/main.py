# -x is 2**63 for x == -2**63, so the divisor is 0. In 64 bits -x is still -2**63.
def u() -> int:
    x = -9223372036854775807 - 1
    return 100 // ((-x) // 2**62 - 2)


u()
