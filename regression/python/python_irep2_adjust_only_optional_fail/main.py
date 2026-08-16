# Negative counterpart: same inline-Optional sole-adjuster path, false assertion,
# so the padded struct literal must reach the solver and report the violation
# rather than crash on the operand-count invariant.
def maybe(flag: bool) -> int | None:
    return None if flag else 42


assert maybe(False) == 7
