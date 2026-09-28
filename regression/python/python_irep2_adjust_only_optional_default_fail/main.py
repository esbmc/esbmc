# Negative counterpart: same defaulted-Optional sole-adjuster path, false
# assertion, so the call must reach the solver and report the violation rather
# than abort on the argument struct-type mismatch.
def use(x: int | None = None) -> int:
    if x is None:
        return 0
    return x


assert use() == 5
