# With `replace` False the result is 7, so the assertion fails.
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
assert a == 42
