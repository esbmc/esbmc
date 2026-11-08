def f(a, b):
    return a + b


r1 = f(1, 2)
r2 = f("x", "y")
assert r2 == "yx"
