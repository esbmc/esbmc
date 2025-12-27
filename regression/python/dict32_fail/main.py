d = {"a": 2}
x = d.get("b", d.get("a"))
assert x == 1
