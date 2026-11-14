# The keyword argument reaches g through h, so h(x=2) is 2.
def g(x=1):
    return x


h = g
assert h(x=2) == 2
