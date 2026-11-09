# #9821: a value a function stores keeps its own type when popped and re-stored.
x = {}
x["k"] = "s"


def f():
    x["j"] = 5


f()
d = x.pop("j")
x["j"] = d
assert d == 5
