def f3():
    return "x"


def f2(a=f3):
    return a()


def f1(a, b):
    return a(b)


c = f1(f2, f3)
assert c == "x"
