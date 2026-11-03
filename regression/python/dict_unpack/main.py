m = {"x": 1}
n = {**m, "y": 2}
assert n["y"] == 2
assert n["x"] == 1
