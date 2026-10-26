def invert(pairs):
    return {v: u for u, v in pairs}


r = invert([("a", "b"), ("c", "d")])
assert r["b"] == "c"
