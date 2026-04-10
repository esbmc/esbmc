d = {"a": "python", "b": "esbmc"}

keys = d.keys()
values = d.values()

assert len(keys) == 2
assert len(values) == 2

for i in keys:
    assert i == "a" or i == "b"

for i in values:
    assert i == "python" or i == "esbmc"
