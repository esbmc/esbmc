def f():
    return "a"


d = {"x": 0}
d["k"] = f
assert d["k"]() == "a"
