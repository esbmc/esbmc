d = {"a": {"x": 1}, "b": {"y": 2}}
for v in d.values():
    assert isinstance(v, dict)
