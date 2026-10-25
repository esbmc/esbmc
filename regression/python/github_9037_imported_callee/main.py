# f2 returns the str that the imported apply returns, not what its argument k
# returns, so f1's parameter holds two functions returning str (#9037).
from helper import apply


def g():
    return "x"


def k():
    return 2


def f2():
    return apply(k)


def f1(a):
    return a()


c = f1(g)
d = f1(f2)
assert c == "x"
assert d == "s"
