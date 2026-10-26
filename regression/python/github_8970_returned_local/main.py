# run holds the function in a local and returns its call, so it dispatches on the local.
def f():
    return "a"


def g():
    return 1


def run(c):
    h = f if c else g
    return h()


assert run(True) == "a"
assert run(False) == 1
n = nondet_bool()
r = run(n)
if n:
    assert isinstance(r, str)
    assert r == "a"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
