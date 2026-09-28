# The same rebinding hazard on a parameter: h() is 99, not the 5 that n held at
# the def. A parameter is only captured when the enclosing body never rebinds
# it (#6256).


def f(n):
    def g():
        return n

    n = 99
    return g


h = f(5)
assert h() == 5
