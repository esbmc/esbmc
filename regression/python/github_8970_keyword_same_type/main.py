# Both functions passed by keyword return an int, so the call is accepted
# (#8970).
def f():
    return 2


def g():
    return 1


def call(h):
    return h()


def run(c):
    if c:
        return call(h=f)
    return call(h=g)


assert run(True) == 2
assert run(False) == 1
