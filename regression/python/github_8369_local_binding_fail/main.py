# A name the function binds is its local, so reading it first raises
# UnboundLocalError in CPython, whatever the module binds (#8369).
def f() -> int:
    x = d
    d = 2
    return x


d = 1
assert f() == 1
