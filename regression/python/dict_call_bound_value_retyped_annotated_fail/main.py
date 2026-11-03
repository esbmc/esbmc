def f():
    w = {"B": 1}
    w["B"] = 2.5
    return w


r = f()
x: int = r["B"]
assert x != 2.5
