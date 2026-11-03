def f():
    w = {"B": 1, "C": 2}
    w["C"] = 3
    return w


def g() -> dict[str, float]:
    return {"B": 2.5}


r = f()
assert r["B"] + r["C"] == 4
x: int = r["C"]
assert x == 3

s = g()
s["B"] = 0.5
assert s["B"] == 0.5
