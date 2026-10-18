d = {k: v for k, v in [("a", 1), ("b", 2)]}
assert len(d) == 2
assert d["b"] == 2

e = {x: x * x for x in range(3)}
assert e[2] == 4
