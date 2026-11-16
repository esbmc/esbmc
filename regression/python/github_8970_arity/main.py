# pick calls ls[key]() with no argument; ls[0] is pick itself, which takes one.
def f():
    return 1


def g():
    return "a"


def pick(key):
    return ls[key]()


ls = [pick, f, g]
key = nondet_int() % 2 + 1
r = pick(key)
if key == 1:
    assert r == 1
else:
    assert r == "a"
