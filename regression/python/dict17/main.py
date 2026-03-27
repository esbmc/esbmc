d = {"a": 10}
assert d.get("a") == 10  # key exists
assert d.get("missing") is None  # key missing; default None
assert d.get("missing", 5) == 5  # custom default returned
