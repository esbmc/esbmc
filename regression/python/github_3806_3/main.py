# list(d.keys()) and list(d.values()) should also work correctly
d: dict[str, int] = {}
assert list(d.keys()) == []
assert list(d.values()) == []
