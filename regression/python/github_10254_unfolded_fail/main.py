# Issue #10254: a folded //= floors, so the truncated quotient is wrong.
def aug(a: int, b: int) -> int:
    a //= b
    return a


assert aug(-7, 2) == -3
