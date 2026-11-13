def f(k: int):
    x = 0
    if k == 1:
        x = "a"
    return x
y = f(1)
z = f(2)
assert y == "a"
assert z == 0
assert isinstance(z, int)
assert not isinstance(y, int)
