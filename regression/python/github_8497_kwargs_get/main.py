def f(x, y=2, **kwargs):
    return x + y + kwargs.get('a', 1)


assert f(1, a=5) == 8
assert f(1, 3) == 5
assert f(1, y=0, b=7) == 2
