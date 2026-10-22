# The functions reach call's parameter by keyword, so the call through it may
# return a str or an int and is refused (#8970).
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


def run(c):
    if c:
        return call(h=f)
    return call(h=g)


assert run(True) == "a"
assert run(False) == 1
