def f(a, b):
    return a + b


r1 = f(1, 2)
r2 = f("x", "y")
assert isinstance(r2, str)
assert r1 == 3
assert r2 == "xy"
