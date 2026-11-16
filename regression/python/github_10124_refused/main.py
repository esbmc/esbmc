# The decorator returns `inner` or `g`, which return different types.
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
    return "Hello"


a = g()
if replace:
    assert isinstance(a, int)
