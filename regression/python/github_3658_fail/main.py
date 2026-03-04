d: dict[str, int] = {"a": 1, "b": 2}
result: int = d.setdefault("d", 4)
assert result == 99  # wrong: setdefault returns the default (4), not 99
