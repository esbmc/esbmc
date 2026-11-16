def pairs_to_dict(g):
    r = {v: 0 for u, v in g}
    return r


def names() -> dict[str, str]:
    r = {"a": "x"}
    return r


def counts():
    r = {}
    r["B"] = 5
    return r


assert pairs_to_dict({("A", "B"): 3})["B"] == 0
assert names()["a"] == "x"
x = counts()["B"]
assert x == 5
assert counts()["B"] + 1 == 6
