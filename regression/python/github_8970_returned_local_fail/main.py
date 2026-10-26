# As github_8970_returned_local, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


def run(c):
    h = f if c else g
    return h()


n = nondet_bool()
r = run(n)
if not n:
    assert r != 1
