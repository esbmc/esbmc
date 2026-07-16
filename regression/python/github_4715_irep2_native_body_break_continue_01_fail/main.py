# _fail sibling of github_4715_irep2_native_body_break_continue_01 (W1-loc
# spike Phase C, esbmc/esbmc#4715). Pins that consuming break/continue
# natively neither corrupts the loop's result nor suppresses bug detection:
# with_continue(5) is 13, so the wrong-value assertion is a reachable
# violation reported as VERIFICATION FAILED under --irep2-native-body.


def with_continue(n: int) -> int:
    s: int = 0
    i: int = 0
    while i < n:
        i = i + 1
        if i == 2:
            continue
        s = s + i
    return s


r: int = with_continue(5)
assert r == 14
