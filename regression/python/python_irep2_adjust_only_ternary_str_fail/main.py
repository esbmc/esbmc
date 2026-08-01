# Negative counterpart: the same string-valued ternary with a false assertion.
# The decayed branch must reach the solver and report the violation, not abort
# the encoding.
def pick(b: bool) -> str:
    s: str = "" if b else "foo"
    return s


assert len(pick(False)) == 0
