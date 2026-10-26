def invert(pairs):
    return {v: u for u, v in pairs}


def index(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


r = invert([("a", "b"), ("c", "d")])
assert r["b"] == "a"
assert r["d"] == "c"
i = index(["x", "y"])
assert i["y"] == 1
assert i["x"] == 0
