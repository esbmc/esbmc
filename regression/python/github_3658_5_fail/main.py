d: dict[str, int] = {}
r: int = d.setdefault("k")
# wrong: asserts key is NOT in d after setdefault — must fail
assert "k" not in d
