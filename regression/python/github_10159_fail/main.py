# A module-level def named len is what len() calls (#10159).
def len(v) -> int:
    return 3


xs = [1]
assert len(xs) == 1
