# As github_8970_arity, but the str path asserts what is false.
def f():
    return 1


def g():
    return "a"


def pick(key):
    return ls[key]()


ls = [pick, f, g]
key = nondet_int() % 2 + 1
r = pick(key)
if key == 2:
    assert r == 1
