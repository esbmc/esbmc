def func2(v="Hello"):
    return v


def func1(*args):
    return func2(*args)


a = func1()
assert isinstance(a, str)
assert a == "Hello"
