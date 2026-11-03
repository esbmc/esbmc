# The functions reach call's keyword-only parameter through an alias of call
# (#8970).
def f():
    return "a"


def g():
    return 1


def call(*, h):
    return h()


c = call
assert c(h=f) == "a"
assert isinstance(c(h=g), int)
