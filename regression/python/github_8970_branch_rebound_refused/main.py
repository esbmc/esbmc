# The rest of the branch rebinds a name that was bound before the if, which is
# not renamed, so the call is refused as before.
def f():
    return "a"


def g():
    return 1


n = 0
h = f if nondet_bool() else g
if nondet_bool():
    r = h()
    n = 1
assert n >= 0
