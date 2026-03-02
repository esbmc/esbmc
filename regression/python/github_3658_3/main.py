d: dict[str, int] = {"x": 10}
# key "x" already present — returns 10, no mutation
r1: int = d.setdefault("x", 0)
assert r1 == 10
assert d["x"] == 10
# key "y" missing — inserts and returns 20
r2: int = d.setdefault("y", 20)
assert r2 == 20
assert d["y"] == 20
