# dict.popitem() with int values: verify sequential removal
d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
k, v = d.popitem()
assert k == "c"
assert v == 3
k2, v2 = d.popitem()
assert k2 == "b"
assert v2 == 2
