def call(g, v):
    return g(v)


a = call(lambda x: x * 2, 3)
assert a == 6
if not isinstance(a, int):
    raise TypeError("int")
