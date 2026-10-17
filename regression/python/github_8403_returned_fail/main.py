# The result is an int, never a float.
def make():
    return lambda x: x * 2


f = make()
a = f(3)
assert isinstance(a, float)
