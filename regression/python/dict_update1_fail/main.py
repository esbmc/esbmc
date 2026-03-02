# dict.update() with new keys; assert on wrong value must fail
d: dict = {"x": 10}
d.update({"y": 20})
assert d["y"] == 99  # wrong: y was set to 20
