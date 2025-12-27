d = {"a": 1}
x = d.get("b", d.get("a"))
assert x == 1
