# Test dict.pop() with annotated result variable and string values
d: dict[str, str] = {"name": "alice", "city": "nyc"}
v: str = d.pop("name")
assert v == "alice"
assert "name" not in d
assert "city" in d
