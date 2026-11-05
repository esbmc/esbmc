# #9821, false assertion: a value a function stores keeps its own type.
x = {}
x["k"] = "s"


def f():
    x["j"] = 5


f()
d = x.pop("j")
x["j"] = d
assert d == 6
