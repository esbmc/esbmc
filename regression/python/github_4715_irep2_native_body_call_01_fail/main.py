# _fail sibling of github_4715_irep2_native_body_call_01 (W1-loc spike Phase
# C, esbmc/esbmc#4715). Pins that consuming bare call statements natively
# neither corrupts the accumulated global nor suppresses bug detection:
# run() returns 5, so the wrong-value assertion is a reachable violation
# reported as VERIFICATION FAILED under --irep2-native-body.

g: int = 0


def bump(x: int) -> None:
    global g
    g = g + x


def reset() -> None:
    global g
    g = 0


def bump_twice(x: int, y: int) -> None:
    bump(x)
    bump(y)


def run() -> int:
    reset()
    bump_twice(2, 3)
    return g


r: int = run()
assert r == 6
