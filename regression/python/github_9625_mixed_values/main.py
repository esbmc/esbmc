# Control: stores of two kinds leave the value type unrecovered, as before.
x = {}
x["a"] = 1
x["b"] = "s"
v = x.get("a")
assert v == 1
