# call(5) is never reached, but the parameter would then take a value that is no function.
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


assert call(f) == "a"
assert call(g) == 1
if False:
    call(5)
