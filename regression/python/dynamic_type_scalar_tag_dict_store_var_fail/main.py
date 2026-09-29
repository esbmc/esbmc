cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
d = {"k": x}
y = d["k"]
assert y == 1 and y == "a"
