def foo(x: int):
    return int.to_bytes(x, 2, "big")

def bar(x: int):
    return int.to_bytes(x, 2, "little")

x = 255
y = foo(x)
assert y[0] == 0
assert y[1] == 255

z = bar(x)
assert z[0] == 255
assert z[1] == 0
