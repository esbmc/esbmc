d = {"properties": {"x": 1}}
for k, v in d["properties"].items():
    assert k == "x"
    assert v == 1
