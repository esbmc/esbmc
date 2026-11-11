# abs(x) is 2**63 for x == -2**63, so the divisor is 0. In 64 bits it stays
# -2**63.
def b() -> int:
    x = -9223372036854775807 - 1
    return 100 // (abs(x) // 2**62 - 2)


b()
