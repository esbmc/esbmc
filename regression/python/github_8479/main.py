def make(k: str, v: int) -> dict:
    return {k: v}


a = {"x": 1, "z": 5}
b = {"y": 2, "x": 3}
m = {**a, "w": 7, **b}
assert m["x"] == 3
assert m["y"] == 2
assert m["z"] == 5
assert m["w"] == 7
assert len(m) == 4
n = a | b
assert n["x"] == 3
assert len(n) == 3
assert a["x"] == 1
assert len(a) == 2
f = make("p", 8) | {**make("q", 9)}
assert f["p"] == 8
assert f["q"] == 9
