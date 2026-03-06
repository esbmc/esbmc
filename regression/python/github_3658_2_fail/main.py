d: dict[str, int] = {"a": 1, "b": 2}
result: int = d.setdefault("a", 99)
# wrong: expects setdefault to overwrite existing key — it should not
assert result == 99
