def f(x: int):
    if x > 0:
        return x
    else:
        return "neg"


def g(x):
    if x > 0:
        return x
    else:
        return "neg"


a = f(-5)
assert a == "neg"
b = g(-5)
assert b == "neg"
c = g(3)
assert c == 3

n = nondet_int()
d = f(n)
if n <= 0:
    assert d == "neg"
else:
    assert d != "neg"
    assert d == n
