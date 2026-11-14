# Issue #10254: a folded call floors //, so the truncated quotient is wrong.
def q(a: int, b: int) -> int:
    return a // b


assert q(0, -3) == 0
assert q(-7, 2) == -3
