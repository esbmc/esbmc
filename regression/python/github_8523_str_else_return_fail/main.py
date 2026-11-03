def f(x: int):
    if x > 0:
        return x
    else:
        return "neg"


a = f(-5)
assert a == "neg"
b = f(3)
assert b == "neg"
