# dict.popitem() reduces dict size and can empty it
d: dict[str, int] = {"x": 10, "y": 20}
k1, v1 = d.popitem()
assert k1 == "y"
assert v1 == 20
k2, v2 = d.popitem()
assert k2 == "x"
assert v2 == 10
