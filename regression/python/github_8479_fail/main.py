a = {"x": 1}
b = {"x": 2}
m = {"x": 0, **a, **b}
n = a | b
assert m["x"] == 2
assert n["x"] == 1
