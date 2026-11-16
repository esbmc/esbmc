def n():
    return 5

def f(a):
    if a == 0:
        return "s"
    else:
        return n()

d = f(1)
assert d == 5
