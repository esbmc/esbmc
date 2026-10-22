# A keyword-only default and a keyword argument return different types
# (#8970).
def f():
    return "a"


def g():
    return 1


def call(*, h=f):
    return h()


assert isinstance(call(h=g), int)
