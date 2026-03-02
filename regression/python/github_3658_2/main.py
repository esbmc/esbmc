d: dict[str, int] = {"a": 1, "b": 2}
result: int = d.setdefault("a", 99)
# key "a" already exists with value 1 — must return 1, not 99
assert result == 1
# dict must remain unchanged
assert d["a"] == 1
