def f(x, y=2, **kwargs):
    return x + y + kwargs.get('a', 1)


assert f(1, b=5) == 8
