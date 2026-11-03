def f():
    return 1


x = lambda h: h()
assert x(f) == 1
