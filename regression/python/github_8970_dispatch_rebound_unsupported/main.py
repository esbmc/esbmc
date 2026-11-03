# r is bound before the call through h and read where it is rebound, so it
# cannot be renamed per function and the call is not dispatched.
def f():
    return "a"


def g():
    return 1


r = 0
h = f if nondet_bool() else g
r = [r, h()]
assert r[0] == 0
