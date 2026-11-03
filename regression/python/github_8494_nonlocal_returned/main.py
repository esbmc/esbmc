def make():
    c = 0
    def inc():
        nonlocal c
        c += 1
        return c
    return inc
a = make()
b = make()
a()
b()
assert a() == 2
