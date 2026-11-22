def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


def run(c):
    return call(h=f if c else g)


assert run(True) == "a"
assert run(False) == 1
