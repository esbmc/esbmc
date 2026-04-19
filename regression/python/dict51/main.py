d = {"a": 1, "b": 2}
keys = d.keys()
for i in keys:
    assert i == "a" or i == "b"
