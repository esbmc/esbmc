# f's call sites disagree on x's kind (num vs str), so x is tagged; the str
# call then raises the same TypeError CPython would on x + 1.
def f(x):
    y = x + 1
    return y


a = f(3)
assert a == 4
b = f("s")
