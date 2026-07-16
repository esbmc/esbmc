# _fail sibling of github_4715_irep2_native_body_while_01 (W1-loc spike Phase
# C, esbmc/esbmc#4715). Pins that consuming the while loop natively neither
# corrupts the accumulated values nor suppresses bug detection: sum_to(5) is
# 10, so the wrong-value assertion is a reachable violation reported as
# VERIFICATION FAILED under --irep2-native-body.


def sum_to(n: int) -> int:
    s: int = 0
    i: int = 0
    while i < n:
        s = s + i
        i = i + 1
    return s


r: int = sum_to(5)
assert r == 11
