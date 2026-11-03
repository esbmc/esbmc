def make() -> dict[str, dict[str, int]]:
    inner = {"t": 1}
    outer = {"k": inner}
    inner["u"] = 2
    return outer


a = make()
b = make()
assert a["k"]["u"] == 2
a["k"]["t"] = 5
assert b["k"]["t"] == 5
