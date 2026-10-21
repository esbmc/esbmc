# The call's result is returned from call, so it leaves the statements that
# could be copied per function.
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


assert call(f) == "a"
assert call(g) == 1
