# The default reaches call's parameter as well as the argument, so the call
# through it may return a str or an int and is refused (#8970).
def f():
    return "a"


def g():
    return 1


def call(h=f):
    return h()


assert isinstance(call(g), int)
