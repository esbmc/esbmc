# The result of an unmodelled call used as a truth value under
# --multi-property (#9208).
import os


def f(p: str) -> bool:
    return not os.path.islink(p)


def g(p: str) -> int:
    if os.path.islink(p):
        return 1
    return 0


def h(p: str) -> int:
    n = 0
    while os.path.islink(p):
        n += 1
        if n > 1:
            break
    return n


b = f("x")
assert b or not b
r = g("x")
assert r == 0 or r == 1
n = h("x")
assert n <= 2
