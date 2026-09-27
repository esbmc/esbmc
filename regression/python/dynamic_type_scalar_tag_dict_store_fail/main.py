cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
d = {"k": x}
assert d["k"] == 1 and d["k"] == "a"
