# The call through h is in a loop body, which is not copied per function.
def f():
    return "a"


def g():
    return 1


h = f if nondet_bool() else g
for _ in range(2):
    r = h()
    assert r == "a" or r == 1
