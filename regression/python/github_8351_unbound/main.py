# `g` is unbound when the branch is not taken, where CPython raises
# NameError, so the call is refused (#8351).
def f1() -> int:
    return 1


if nondet_bool():
    g = f1
r = g()
assert r == 1
