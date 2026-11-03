# The lambda make returns is called through f with an int, so its result is
# an int.
def make():
    return lambda x: x * 2


f = make()
a = f(3)
assert a == 6
assert isinstance(a, int)
