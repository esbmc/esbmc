# A call through an element of a list of functions (#8351).
def f1() -> int:
    return 1


fs = [f1]
g = fs[0]
r = g()
assert r == 2
