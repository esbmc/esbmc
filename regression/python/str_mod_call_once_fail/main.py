# A call operand of str % runs once, and a tuple's operands are evaluated
# left to right.
n = 0
w = "w"


def a() -> int:
    global n, w
    n = n * 10 + 1
    w = "x"
    return n


def b() -> str:
    global n
    n = n * 10 + 2
    return "b"


def c(k: int) -> str:
    t = "z"
    if k > 0:
        t = "%s" % b()
    return t


s = "%s%d%s" % (w, a(), b())
assert n == 21
assert s == "w1b"
assert "%d" % a() == "121"
assert n == 121
r = c(1)
assert r == "b"
assert n == 1212
