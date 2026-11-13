# h(x=2) is 2, not the default 1.
def g(x=1):
    return x


h = g
assert h(x=2) == 1
