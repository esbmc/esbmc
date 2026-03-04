d: dict[str, int] = {"x": 10}
r1: int = d.setdefault("x", 0)
# wrong: "x" was already present so result must be 10, not 0
assert r1 == 0
