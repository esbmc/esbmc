def f():
    w = {"B": 2.5}
    w["B"] = 1
    return w


r = f()
assert r["B"] != 1
