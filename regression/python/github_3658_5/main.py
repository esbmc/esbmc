d: dict[str, int] = {}
r: int = d.setdefault("k")
# key must have been inserted even when no default is given
assert "k" in d
