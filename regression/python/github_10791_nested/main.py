d = {"a": {"b": {"c": 1, "e": 2.5, "l": [1]}}}
d["a"]["b"]["l"].append(2)
assert len(d["a"]["b"]["l"]) == 2
assert d["a"]["b"]["c"] == 1
assert d["a"]["b"]["e"] == 2.5
