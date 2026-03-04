d: dict[str, int] = {"a": 1, "b": 2}
result: int = d.setdefault("d", 4)
assert result == 4
assert d["d"] == 4
