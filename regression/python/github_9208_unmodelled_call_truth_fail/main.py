# The result of an unmodelled call used as a truth value under
# --multi-property (#9208).
import os


def f(p: str) -> bool:
    return not os.path.islink(p)


def g(p: str) -> int:
    if os.path.islink(p):
        return 1
    return 0


assert not f("x")
assert g("x") == 1
