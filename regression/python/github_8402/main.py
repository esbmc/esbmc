a = 1
if not isinstance(a, int):
    raise TypeError("int")
a = 1.5
assert isinstance(a, float)
assert a == 1.5
