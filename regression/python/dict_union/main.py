a = {"a": 1}
b = {"b": 2}
c = a | b
assert c["b"] == 2
assert c["a"] == 1
