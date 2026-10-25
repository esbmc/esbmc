# Issue #9156: a def nested under an if in another def returns the enclosing
# local, not the module-level name holding a str or an int.
c = nondet_bool()
if c:
    r = "a"
else:
    r = 1


def f():
    r = 5
    if True:

        def g():
            return r

    return g()


z = f()
assert not isinstance(z, int)
