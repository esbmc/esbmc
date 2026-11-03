def f3():
    return "x"


def f2(a=f3):
    return a()


c = f2(f2)
assert c == "x"
