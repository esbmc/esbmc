f = lambda x: x + 1
a = f(1)
b = f(2.5)
assert isinstance(a, int)
assert a == 2
assert isinstance(b, float)
assert b == 3.5


def make():
    return lambda x: x * 2


g = make()
c = g(3)
d = g(0.5)
assert isinstance(c, int)
assert c == 6
assert isinstance(d, float)
