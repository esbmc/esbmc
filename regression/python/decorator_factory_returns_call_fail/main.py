def factory(replace):
    def dec(f):
        if replace:
            return inner()
        return f

    def inner():
        return 42

    return dec


replace = True


@factory(replace)
def g():
    return "g"


a = g()
assert a == "g"
