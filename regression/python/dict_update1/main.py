# dict.update() inserts new key-value pairs
d: dict = {"x": 10}
d.update({"y": 20, "z": 30})
assert d["x"] == 10
assert d["y"] == 20
assert d["z"] == 30
