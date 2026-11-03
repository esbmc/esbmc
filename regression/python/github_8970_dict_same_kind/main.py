# Both functions return an int, so nothing is dispatched.
def inc():
    return 1


def dbl():
    return 2


d = {"a": inc, "b": dbl}
c = nondet_bool()
r = d["a" if c else "b"]()
assert r == (1 if c else 2)
