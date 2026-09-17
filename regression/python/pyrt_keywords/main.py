def f(a, b=2):
    return a + b


assert f(a=1) == 3
assert f(b=5, a=1) == 6
assert f(1, b=5) == 6
