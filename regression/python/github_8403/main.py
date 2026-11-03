f = lambda x: x * 2
a = f(3)
assert a == 6
if not isinstance(a, int):
    raise TypeError("int")
