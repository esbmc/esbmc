def ok1():
    all(1 % x for x in [])


def ok2():
    xs = []
    all(1 % x for x in xs)


ok1()
ok2()
