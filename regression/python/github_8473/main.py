def f():
    return 1


def g():
    return 2


class C:
    pass


calls = 0


def make():
    global calls
    calls += 1
    return C()


def pick():
    return g


b = f
assert b is f
assert f is b
assert f is f
assert not (f is g)
assert f is not g
assert b is not g
c = b
assert c is f
b = g
assert b is g
assert b is not f
assert c is f
assert not (f is None)
assert f is not None
s = "s"
assert f is not s
inst = C()
assert not (f is inst)
assert inst is not f
assert make() is not f
assert calls == 1
h = pick()
assert h is g
assert h is not f
ls = [f, g]
assert ls[0] is f
assert ls[1] is not f
