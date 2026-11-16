# Both functions return an int, so the call is not refused.
def factory(replace):
    def dec(f):
        if replace:
            return inner
        return f

    def inner():
        return 42

    return dec


replace = nondet_bool()


@factory(replace)
def g():
    return 7


a = g()
if replace:
    assert a == 42
else:
    assert a == 7
